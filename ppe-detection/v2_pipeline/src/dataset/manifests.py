"""Build deterministic PPE split manifests without copying source images."""

from __future__ import annotations

import hashlib
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


CLASS_COUNT_COLUMNS = {
    0: "num_person",
    1: "num_helmet",
    2: "num_vest",
    3: "num_cleaning_coverall",
}
STRATIFY_CLASS_IDS = (1, 2, 3)


@dataclass(frozen=True, slots=True)
class YoloSample:
    """One strict image-label pair and its class-presence metadata."""

    source: str
    image_path: Path
    label_path: Path
    class_counts: dict[int, int]

    @property
    def stratify_key(self) -> str:
        """Return a stable helmet/vest/coverall presence signature."""
        return "|".join(
            f"{class_id}={int(self.class_counts.get(class_id, 0) > 0)}"
            for class_id in STRATIFY_CLASS_IDS
        )


def collect_yolo_samples(
    source_dir: Path,
    source_name: str,
    class_names: dict[int, str],
    image_extensions: Iterable[str],
    *,
    allow_empty_labels: bool = False,
    required: bool = True,
) -> list[YoloSample]:
    """Collect strict image-label pairs from one immutable source lane.

    Args:
        source_dir: Source root containing sibling `images/` and `labels/`.
        source_name: Source lane recorded in split assignments.
        class_names: Allowed class map.
        image_extensions: Accepted image suffixes.
        allow_empty_labels: Whether zero-object labels are accepted.
        required: Whether a missing or empty source is an error.

    Returns:
        Samples sorted by resolved image path.

    Raises:
        FileNotFoundError: If required source folders are absent.
        ValueError: If pairs or labels violate the strict data contract.
    """
    source_root = Path(source_dir).resolve()
    images_dir = source_root / "images"
    labels_dir = source_root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        if required:
            raise FileNotFoundError(f"{source_name} requires images/ and labels/: {source_root}")
        return []

    extensions = {str(extension).lower() for extension in image_extensions}
    images_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            images_by_stem[path.stem].append(path.resolve())
    labels_by_stem = {
        path.stem: path.resolve()
        for path in sorted(labels_dir.iterdir())
        if path.is_file() and path.suffix.lower() == ".txt"
    }

    duplicate_stems = sorted(stem for stem, paths in images_by_stem.items() if len(paths) > 1)
    if duplicate_stems:
        raise ValueError(f"Duplicate image stems in {source_name}: {duplicate_stems[:10]}")

    image_stems = set(images_by_stem)
    label_stems = set(labels_by_stem)
    if image_stems != label_stems:
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        raise ValueError(
            f"{source_name} image-label mismatch; missing labels={missing_labels[:10]}, "
            f"missing images={missing_images[:10]}"
        )
    if required and not image_stems:
        raise ValueError(f"{source_name} has no image-label pairs: {source_root}")

    samples = []
    for stem in sorted(image_stems):
        image_path = images_by_stem[stem][0]
        label_path = labels_by_stem[stem]
        class_counts = _parse_label_counts(
            label_path,
            allowed_class_ids=set(class_names),
            allow_empty=allow_empty_labels,
        )
        samples.append(
            YoloSample(
                source=source_name,
                image_path=image_path,
                label_path=label_path,
                class_counts=class_counts,
            )
        )
    return samples


def build_split_assignments(
    open_source_samples: list[YoloSample],
    factory_samples: list[YoloSample],
    test_samples: list[YoloSample],
    *,
    factory_train_ratio: float,
    factory_val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign source samples to train, validation, and untouched test splits.

    Args:
        open_source_samples: Train-only external samples.
        factory_samples: Target-domain samples to stratify into train/val.
        test_samples: Independently supplied final-test samples.
        factory_train_ratio: Factory training fraction.
        factory_val_ratio: Factory validation fraction.
        seed: Deterministic group shuffle seed.

    Returns:
        `(assignments, warnings)` DataFrames.

    Raises:
        ValueError: If ratios are invalid, a required split is empty, or split
            membership overlaps.
    """
    _validate_ratios(factory_train_ratio, factory_val_ratio)
    if not factory_samples:
        raise ValueError("factory_source must contain valid samples")
    if not test_samples:
        raise ValueError("test_source must contain valid samples")

    factory_train, factory_val, warning_rows = _stratified_factory_split(
        factory_samples,
        val_ratio=factory_val_ratio,
        seed=seed,
    )
    split_plan = {
        "train": [*open_source_samples, *factory_train],
        "val": factory_val,
        "test": test_samples,
    }
    rows = [
        _assignment_row(sample, split_name)
        for split_name, samples in split_plan.items()
        for sample in samples
    ]
    assignments = pd.DataFrame(rows).sort_values(
        ["split", "source", "image_path"], kind="stable"
    ).reset_index(drop=True)
    validate_split_assignments(assignments)
    # Missing split-level class coverage is a modeling limitation worth
    # reviewing, but it should not silently reshuffle an independent test set.
    for split_name in ("train", "val", "test"):
        split_rows = assignments.loc[assignments["split"].eq(split_name)]
        for class_id, count_column in CLASS_COUNT_COLUMNS.items():
            if int(split_rows[count_column].sum()) == 0:
                warning_rows.append(
                    {
                        "warning_type": "missing_class_coverage",
                        "details": f"split={split_name} has no class_id={class_id} objects",
                    }
                )
    warnings = pd.DataFrame(warning_rows, columns=["warning_type", "details"])
    return assignments, warnings


def validate_split_assignments(assignments: pd.DataFrame) -> None:
    """Enforce source policy and prevent train/validation/test path overlap.

    Args:
        assignments: Output from :func:`build_split_assignments`.

    Raises:
        ValueError: If a split is empty, a path appears in several splits, or a
            source is assigned to a forbidden split.
    """
    required_splits = {"train", "val", "test"}
    present_splits = set(assignments.get("split", pd.Series(dtype=str)))
    missing = required_splits - present_splits
    if missing:
        raise ValueError(f"Required split(s) are empty: {sorted(missing)}")

    overlap = assignments.groupby("image_path")["split"].nunique()
    if (overlap > 1).any():
        paths = overlap[overlap > 1].index.tolist()
        raise ValueError(f"Image paths overlap across splits: {paths[:10]}")

    invalid_open = assignments[
        assignments["source"].eq("open_source") & ~assignments["split"].eq("train")
    ]
    invalid_val = assignments[
        assignments["split"].eq("val") & ~assignments["source"].eq("factory_source")
    ]
    invalid_test = assignments[
        assignments["split"].eq("test") & ~assignments["source"].eq("test_source")
    ]
    if not invalid_open.empty or not invalid_val.empty or not invalid_test.empty:
        raise ValueError("Source policy violation in split assignments")


def summarize_assignments(
    assignments: pd.DataFrame,
    class_names: dict[int, str],
) -> pd.DataFrame:
    """Return one compact count row per split.

    Args:
        assignments: Frozen assignment rows.
        class_names: Fixed class map.

    Returns:
        Split-level image, object, and source counts.
    """
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        split_df = assignments[assignments["split"].eq(split_name)]
        row: dict[str, Any] = {
            "split": split_name,
            "num_images": int(len(split_df)),
            "num_objects": int(split_df["num_objects"].sum()),
            "sources": ", ".join(sorted(split_df["source"].unique())),
        }
        for class_id, class_name in class_names.items():
            row[f"num_{class_name}"] = int(split_df[CLASS_COUNT_COLUMNS[class_id]].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write_split_manifests(
    assignments: pd.DataFrame,
    manifests_dir: Path,
) -> dict[str, Path]:
    """Write frozen split membership and its compact assignment audit.

    Args:
        assignments: Validated assignment rows.
        manifests_dir: Run-local output directory.

    Returns:
        Mapping for `train_original`, `val`, `test`, and `assignments` paths.
    """
    validate_split_assignments(assignments)
    output_dir = Path(manifests_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_original": output_dir / "train_original.txt",
        "val": output_dir / "val.txt",
        "test": output_dir / "test.txt",
        "assignments": output_dir / "split_assignments.csv",
    }
    for split_name, manifest_key in (
        ("train", "train_original"),
        ("val", "val"),
        ("test", "test"),
    ):
        write_manifest(
            paths[manifest_key],
            assignments.loc[assignments["split"].eq(split_name), "image_path"],
        )
    assignments.to_csv(paths["assignments"], index=False)
    return paths


def write_manifest(path: Path, image_paths: Iterable[str | Path]) -> Path:
    """Write sorted, unique absolute image paths to a text manifest.

    Args:
        path: Destination text file.
        image_paths: Image paths to freeze.

    Returns:
        Destination path.

    Raises:
        ValueError: If a duplicate path is supplied.
    """
    resolved = [str(Path(image_path).resolve()) for image_path in image_paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"Duplicate paths cannot be written to manifest: {path}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(f"{item}\n" for item in sorted(resolved)), encoding="utf-8")
    return destination


def read_manifest(path: Path, *, require_files: bool = True) -> list[Path]:
    """Read one image manifest and optionally verify every file.

    Args:
        path: Manifest file.
        require_files: Whether missing image paths are errors.

    Returns:
        Resolved image paths in manifest order.

    Raises:
        ValueError: If the manifest contains duplicate or missing paths.
    """
    manifest_path = Path(path)
    image_paths = [
        Path(line.strip()).resolve()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(image_paths) != len(set(image_paths)):
        raise ValueError(f"Manifest contains duplicate paths: {manifest_path}")
    if require_files:
        missing = [str(image_path) for image_path in image_paths if not image_path.is_file()]
        if missing:
            raise ValueError(f"Manifest references missing images: {missing[:10]}")
    return image_paths


def build_combined_training_manifest(
    original_manifest: Path,
    augmented_images_dir: Path,
    output_path: Path,
    image_extensions: Iterable[str],
) -> Path:
    """Combine original and real offline-augmented image paths without copies.

    Args:
        original_manifest: Frozen original training membership.
        augmented_images_dir: Directory containing generated augmentation images.
        output_path: Destination combined manifest.
        image_extensions: Accepted image suffixes.

    Returns:
        Combined manifest path.

    Raises:
        ValueError: If no augmented images exist.
    """
    original_paths = read_manifest(original_manifest)
    extensions = {str(extension).lower() for extension in image_extensions}
    augmented_paths = (
        sorted(
            path.resolve()
            for path in Path(augmented_images_dir).iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
        if Path(augmented_images_dir).is_dir()
        else []
    )
    if not augmented_paths:
        raise ValueError(f"No offline augmented images found: {augmented_images_dir}")
    return write_manifest(output_path, [*original_paths, *augmented_paths])


def build_refit_training_manifest(
    train_manifest: Path,
    val_manifest: Path,
    output_path: Path,
) -> Path:
    """Combine a selected training manifest with frozen validation membership.

    This final-refit helper preserves whichever training policy was selected:
    original-only manifests remain original-only, while offline-augmentation
    manifests retain their generated samples. Validation images are added as
    clean development examples and the independent test lane remains absent.

    Args:
        train_manifest: Selected original or original-plus-offline membership.
        val_manifest: Frozen clean validation membership.
        output_path: Destination train-plus-validation manifest.

    Returns:
        Sorted, unique final-refit training manifest.

    Raises:
        ValueError: If training and validation membership overlap.
    """
    train_paths = read_manifest(train_manifest)
    val_paths = read_manifest(val_manifest)
    overlap = set(train_paths) & set(val_paths)
    if overlap:
        raise ValueError(
            "Selected training and validation manifests overlap: "
            f"{[str(path) for path in sorted(overlap)[:10]]}"
        )
    return write_manifest(output_path, [*train_paths, *val_paths])


def build_development_training_manifest(
    assignments_csv: Path,
    output_path: Path,
    *,
    include_open_source: bool,
) -> Path:
    """Build one original-data training manifest from frozen train and val rows.

    This helper supports a deliberate final-refit workflow where every
    development sample is used for fitting and the independent test lane stays
    untouched. Offline-generated images are not present in Notebook 02's split
    assignments and are therefore never added implicitly.

    Args:
        assignments_csv: Notebook 02's frozen `split_assignments.csv`.
        output_path: Destination for the combined development manifest.
        include_open_source: Whether train-only `open_source` rows are retained.

    Returns:
        A sorted, unique manifest containing train and validation membership.

    Raises:
        ValueError: If the assignment contract is invalid, an unexpected source
            enters the development pool, or filtering leaves no training data.
    """
    return build_original_training_manifest(
        assignments_csv,
        output_path,
        include_open_source=include_open_source,
        include_validation=True,
    )


def build_original_training_manifest(
    assignments_csv: Path,
    output_path: Path,
    *,
    include_open_source: bool,
    include_validation: bool,
) -> Path:
    """Build a source-filtered original-data training manifest.

    Args:
        assignments_csv: Notebook 02's frozen `split_assignments.csv`.
        output_path: Destination manifest path.
        include_open_source: Whether train-only `open_source` rows are retained.
        include_validation: Whether frozen factory validation rows join the
            training rows for a final refit. False produces the calibration
            training membership.

    Returns:
        A sorted, unique manifest containing the requested original samples.

    Raises:
        ValueError: If the assignment contract is invalid, an unexpected source
            enters training, or filtering leaves no data.
    """
    assignments_path = Path(assignments_csv)
    assignments = pd.read_csv(assignments_path)
    required_columns = {"split", "source", "image_path"}
    missing_columns = required_columns - set(assignments.columns)
    if missing_columns:
        raise ValueError(
            f"Split assignments are missing columns: {sorted(missing_columns)}"
        )

    # Validate the complete frozen contract before selecting development rows;
    # this catches accidental source-policy or cross-split changes upstream.
    validate_split_assignments(assignments)
    selected_splits = ("train", "val") if include_validation else ("train",)
    training_rows = assignments.loc[
        assignments["split"].isin(selected_splits)
    ].copy()
    unexpected_sources = set(training_rows["source"]) - {
        "open_source",
        "factory_source",
    }
    if unexpected_sources:
        raise ValueError(
            "Unexpected source(s) in development membership: "
            f"{sorted(unexpected_sources)}"
        )
    if not include_open_source:
        training_rows = training_rows.loc[
            ~training_rows["source"].eq("open_source")
        ]
    if training_rows.empty:
        raise ValueError("Original training membership is empty after filtering")

    destination = write_manifest(output_path, training_rows["image_path"])
    # Resolve every image immediately so a stale assignment cannot fail only
    # after a long model initialization step.
    read_manifest(destination)
    return destination


def copy_manifest(source: Path, destination: Path) -> Path:
    """Copy one small manifest into a run for a self-contained record."""
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(source), destination_path)
    return destination_path


def manifest_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a frozen manifest file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_pair_content_sha256(path: Path) -> str:
    """Hash the ordered image and label bytes referenced by one manifest.

    Args:
        path: Image manifest whose sibling YOLO labels must exist.

    Returns:
        One digest that changes when any referenced image or label changes.

    Raises:
        FileNotFoundError: If a sibling label is missing.

    Notes:
        Absolute paths are intentionally excluded so identical data moved to a
        different machine retains the same content fingerprint.
    """
    digest = hashlib.sha256()
    for image_path in read_manifest(path):
        label_path = image_to_label_path(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label referenced by manifest: {label_path}")
        for role, content_path in ((b"image", image_path), (b"label", label_path)):
            digest.update(role)
            file_size = content_path.stat().st_size
            digest.update(file_size.to_bytes(8, byteorder="big", signed=False))
            with content_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def image_to_label_path(image_path: Path) -> Path:
    """Resolve an image to its sibling YOLO label path.

    Args:
        image_path: Image beneath an `images/` directory.

    Returns:
        Matching path beneath the sibling `labels/` directory.

    Raises:
        ValueError: If the image path has no `images` directory component.
    """
    parts = list(Path(image_path).resolve().parts)
    image_indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not image_indexes:
        raise ValueError(f"YOLO image path has no images directory: {image_path}")
    parts[image_indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def write_ultralytics_dataset_yaml(
    output_path: Path,
    train_manifest: Path,
    val_manifest: Path,
    class_names: dict[int, str],
    *,
    test_manifest: Path | None = None,
    allow_train_val_overlap: bool = False,
) -> Path:
    """Write one run-local Ultralytics dataset YAML from frozen manifests.

    Args:
        output_path: Run-local YAML destination.
        train_manifest: Training image manifest.
        val_manifest: Validation image manifest.
        class_names: Fixed class map.
        test_manifest: Optional untouched-test manifest. Candidate and ablation
            callers must omit it.
        allow_train_val_overlap: Explicitly permit a final-refit dataset to use
            the same full development manifest for both required Ultralytics
            keys. Callers must disable validation and early stopping while
            fitting; the test manifest remains disjoint.

    Returns:
        Dataset YAML path.

    Raises:
        ValueError: If supplied manifests overlap.
    """
    train_paths = set(read_manifest(train_manifest))
    val_paths = set(read_manifest(val_manifest))
    if train_paths & val_paths and not allow_train_val_overlap:
        raise ValueError("Training and validation manifests overlap")

    payload: dict[str, Any] = {
        "train": str(Path(train_manifest).resolve()),
        "val": str(Path(val_manifest).resolve()),
        "nc": len(class_names),
        "names": {int(class_id): name for class_id, name in class_names.items()},
    }
    if test_manifest is not None:
        test_paths = set(read_manifest(test_manifest))
        if train_paths & test_paths or val_paths & test_paths:
            raise ValueError("Final test manifest overlaps train or validation")
        payload["test"] = str(Path(test_manifest).resolve())

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return destination


def experiment_manifest(
    experiment_spec: dict[str, Any],
    original_manifest: Path,
    original_plus_offline_manifest: Path,
) -> Path:
    """Resolve an experiment specification to one of two training manifests.

    Args:
        experiment_spec: Validated experiment registry entry.
        original_manifest: Original-only training manifest.
        original_plus_offline_manifest: Combined training manifest.

    Returns:
        Selected manifest path.

    Raises:
        ValueError: If the dataset variant is unknown.
    """
    variant = experiment_spec.get("dataset_variant")
    if variant == "original":
        return Path(original_manifest)
    if variant == "original_plus_offline":
        return Path(original_plus_offline_manifest)
    raise ValueError(f"Unknown dataset variant: {variant}")


def _stratified_factory_split(
    samples: list[YoloSample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[YoloSample], list[YoloSample], list[dict[str, str]]]:
    """Split factory samples by PPE class-presence groups deterministically."""
    grouped: dict[str, list[YoloSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.stratify_key].append(sample)

    rng = random.Random(seed)
    train: list[YoloSample] = []
    val: list[YoloSample] = []
    warnings: list[dict[str, str]] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda sample: str(sample.image_path))
        rng.shuffle(group)
        if len(group) == 1:
            train.extend(group)
            warnings.append(
                {
                    "warning_type": "stratification_singleton",
                    "details": f"{key} has one factory sample and cannot appear in both splits",
                }
            )
            continue
        val_count = min(max(1, round(len(group) * val_ratio)), len(group) - 1)
        val.extend(group[:val_count])
        train.extend(group[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val, warnings


def _assignment_row(sample: YoloSample, split_name: str) -> dict[str, Any]:
    """Convert one sample into a compact frozen assignment row."""
    return {
        "split": split_name,
        "source": sample.source,
        "image_path": str(sample.image_path),
        "label_path": str(sample.label_path),
        "stratify_key": sample.stratify_key,
        "num_objects": sum(sample.class_counts.values()),
        **{
            column: int(sample.class_counts.get(class_id, 0))
            for class_id, column in CLASS_COUNT_COLUMNS.items()
        },
    }


def _parse_label_counts(
    label_path: Path,
    *,
    allowed_class_ids: set[int],
    allow_empty: bool,
) -> dict[int, int]:
    """Parse one strict YOLO label file into class counts."""
    lines = [
        line.strip()
        for line in Path(label_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines and not allow_empty:
        raise ValueError(f"Empty labels are disabled: {label_path}")

    counts = {class_id: 0 for class_id in allowed_class_ids}
    for line_number, line in enumerate(lines, start=1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number} must contain five values")
        try:
            class_value, x_center, y_center, width, height = map(float, values)
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number} contains non-numeric values") from exc
        if not class_value.is_integer() or int(class_value) not in allowed_class_ids:
            raise ValueError(f"{label_path}:{line_number} has invalid class ID {values[0]}")
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
            raise ValueError(f"{label_path}:{line_number} has coordinates outside [0, 1]")
        if width <= 0 or height <= 0:
            raise ValueError(f"{label_path}:{line_number} has non-positive box size")
        if (
            x_center - width / 2 < -1e-4
            or x_center + width / 2 > 1.0 + 1e-4
            or y_center - height / 2 < -1e-4
            or y_center + height / 2 > 1.0 + 1e-4
        ):
            raise ValueError(f"{label_path}:{line_number} box crosses image boundaries")
        counts[int(class_value)] += 1
    return counts


def _validate_ratios(train_ratio: float, val_ratio: float) -> None:
    """Validate positive factory train/validation ratios that sum to one."""
    if train_ratio <= 0 or val_ratio <= 0 or abs(train_ratio + val_ratio - 1.0) > 1e-9:
        raise ValueError("Factory train and validation ratios must be positive and sum to 1")
