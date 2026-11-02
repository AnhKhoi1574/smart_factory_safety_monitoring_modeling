"""Validation helpers for PPE YOLO input sources.

Notebook 01 validates the three source lanes directly:

``open_source``
    Public or external perspective data. It is train-only later.
``factory_source``
    Target-domain CCTV data. It is split into train and validation later.
``test_source``
    Final untouched test data. It is frozen in a manifest and used only during
    final evaluation.

The helpers do not copy, rename, split, augment, or delete files.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd


CLASS_COUNT_COLUMNS = {
    0: "num_person",
    1: "num_helmet",
    2: "num_vest",
    3: "num_cleaning_coverall",
}

VALIDATION_COLUMNS = [
    "source",
    "base_name",
    "image_name",
    "label_name",
    "image_path",
    "label_path",
    "status",
    "errors",
    "warnings",
    "image_width",
    "image_height",
    "num_objects",
    *CLASS_COUNT_COLUMNS.values(),
]

SOURCE_SUMMARY_COLUMNS = [
    "source",
    "total_rows",
    "valid_rows",
    "invalid_rows",
    "warning_rows",
    "total_objects",
    *CLASS_COUNT_COLUMNS.values(),
]

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_CLASS_IDS = {0, 1, 2, 3}
BOUNDARY_TOLERANCE = 1e-4


@dataclass(slots=True)
class LabelStats:
    """Parsed object counts from one YOLO label file."""

    num_objects: int = 0
    class_counts: dict[int, int] | None = None

    def __post_init__(self) -> None:
        """Initialize zero counts for all four classes when omitted."""
        if self.class_counts is None:
            self.class_counts = {class_id: 0 for class_id in DEFAULT_CLASS_IDS}


def validate_input_sources(
    source_dirs: dict[str, Path],
    class_ids: set[int] | None = None,
    image_extensions: set[str] | None = None,
    allow_empty_labels: bool = False,
) -> pd.DataFrame:
    """Validate all named PPE input sources and return one combined report.

    This is the main Notebook 01 entry point. The function intentionally only
    reads files and builds a report; it does not copy files into a canonical
    dataset, rename samples, split train/validation/test, or create generated
    data. Keeping this step read-only makes it safe to rerun whenever new input
    files are pasted into the source folders.

    Args:
        source_dirs: Mapping from source name to a directory containing
            ``images/`` and ``labels/``.
        class_ids: Allowed YOLO class IDs. Defaults to the four-class PPE map.
        image_extensions: Supported image extensions. Defaults to jpg/jpeg/png.
        allow_empty_labels: Whether empty label files are accepted as valid
            no-object images. The default keeps PPE source data strict.

    Returns:
        A validation report with one row per image or unmatched label.
    """
    allowed_class_ids = set(class_ids or DEFAULT_CLASS_IDS)
    supported_extensions = {
        extension.lower() for extension in (image_extensions or VALID_IMAGE_EXTENSIONS)
    }

    reports = [
        validate_source_directory(
            source_dir=source_dir,
            source_name=source_name,
            class_ids=allowed_class_ids,
            image_extensions=supported_extensions,
            allow_empty_labels=allow_empty_labels,
        )
        for source_name, source_dir in source_dirs.items()
    ]
    reports = [report for report in reports if not report.empty]
    if not reports:
        return _empty_report()

    combined_report = pd.concat(reports, ignore_index=True)
    return annotate_cross_source_findings(combined_report)


def build_source_summary(
    validation_df: pd.DataFrame,
    source_names: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Build a compact source-level validation summary for Notebook 01.

    The summary is deliberately small: it answers whether each source lane has
    valid samples, invalid samples, warnings, and enough objects per class to
    proceed to the split notebook. Detailed EDA belongs in later notebooks.

    Args:
        validation_df: Output from :func:`validate_input_sources`.
        source_names: Source names to keep in a stable display order.

    Returns:
        One row per source with row counts and four-class object counts.
    """
    if validation_df.empty:
        return pd.DataFrame(
            [
                {
                    "source": source_name,
                    "total_rows": 0,
                    "valid_rows": 0,
                    "invalid_rows": 0,
                    "warning_rows": 0,
                    "total_objects": 0,
                    **{column: 0 for column in CLASS_COUNT_COLUMNS.values()},
                }
                for source_name in source_names
            ],
            columns=SOURCE_SUMMARY_COLUMNS,
        )

    summary_rows: list[dict[str, Any]] = []
    for source_name in source_names:
        source_df = validation_df.loc[validation_df["source"].eq(source_name)]
        valid_source_df = source_df.loc[source_df["status"].eq("valid")]
        summary_rows.append(
            {
                "source": source_name,
                "total_rows": int(len(source_df)),
                "valid_rows": int(len(valid_source_df)),
                "invalid_rows": int(source_df["status"].ne("valid").sum()),
                "warning_rows": int(source_df["warnings"].fillna("").ne("").sum()),
                "total_objects": int(valid_source_df["num_objects"].sum()),
                **{
                    column: int(valid_source_df[column].sum())
                    for column in CLASS_COUNT_COLUMNS.values()
                },
            }
        )

    return pd.DataFrame(summary_rows, columns=SOURCE_SUMMARY_COLUMNS)


def validate_source_directory(
    source_dir: Path,
    source_name: str,
    class_ids: set[int] | None = None,
    image_extensions: set[str] | None = None,
    allow_empty_labels: bool = False,
) -> pd.DataFrame:
    """Validate one source directory with direct ``images/`` and ``labels/``.

    The expected folder shape is:

    ``source_dir/images/<sample>.jpg``
    ``source_dir/labels/<sample>.txt``

    Every image must have exactly one matching label file with the same base
    name. Empty labels are invalid by default for this PPE workflow because the
    detector is trained on target objects, not background-only samples, unless a
    future notebook explicitly enables no-object images.

    Args:
        source_dir: Source root such as ``data/input/factory_source``.
        source_name: Name to record in reports.
        class_ids: Allowed YOLO class IDs.
        image_extensions: Supported image extensions.
        allow_empty_labels: Whether empty labels are valid.

    Returns:
        A validation report for this source.
    """
    source_dir = Path(source_dir)
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    allowed_class_ids = set(class_ids or DEFAULT_CLASS_IDS)
    supported_extensions = {
        extension.lower() for extension in (image_extensions or VALID_IMAGE_EXTENSIONS)
    }

    rows: list[dict[str, Any]] = []
    if not source_dir.exists():
        rows.append(
            _build_report_row(
                source=source_name,
                base_name="",
                image_path=None,
                label_path=None,
                errors=[f"Source directory does not exist: {source_dir}"],
            )
        )
        return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)

    missing_parts = []
    if not images_dir.exists():
        missing_parts.append("images/")
    if not labels_dir.exists():
        missing_parts.append("labels/")
    if missing_parts:
        rows.append(
            _build_report_row(
                source=source_name,
                base_name="",
                image_path=None,
                label_path=None,
                errors=[f"Missing required folder(s): {', '.join(missing_parts)}"],
            )
        )
        return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)

    # Collect image and label basenames separately. We compare these maps below
    # so missing labels and orphan labels are reported as normal rows instead of
    # causing a hard failure.
    image_paths = _collect_image_paths(images_dir, supported_extensions)
    unsupported_images = _collect_unsupported_image_paths(
        images_dir,
        supported_extensions,
    )
    label_paths = _collect_label_paths(labels_dir)
    duplicate_base_names = {
        base_name for base_name, paths in image_paths.items() if len(paths) > 1
    }

    for base_name in sorted(set(image_paths) | set(label_paths)):
        images_for_base_name = image_paths.get(base_name, [])
        label_path = label_paths.get(base_name)
        if images_for_base_name:
            for image_path in images_for_base_name:
                rows.append(
                    _validate_sample(
                        source=source_name,
                        base_name=base_name,
                        image_path=image_path,
                        label_path=label_path,
                        allowed_class_ids=allowed_class_ids,
                        duplicate_in_source=base_name in duplicate_base_names,
                        allow_empty_labels=allow_empty_labels,
                    )
                )
        else:
            rows.append(
                _build_report_row(
                    source=source_name,
                    base_name=base_name,
                    image_path=None,
                    label_path=label_path,
                    errors=["Label file has no matching image"],
                )
            )

    for image_path in unsupported_images:
        # Ignore placeholders and hidden metadata files. They are useful for Git
        # but should never appear as invalid dataset samples.
        if image_path.name.startswith("."):
            continue
        rows.append(
            _build_report_row(
                source=source_name,
                base_name=image_path.stem,
                image_path=image_path,
                label_path=None,
                errors=[f"Unsupported image extension: {image_path.suffix}"],
            )
        )

    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def annotate_cross_source_findings(report: pd.DataFrame) -> pd.DataFrame:
    """Add warnings for cross-source base-name and content duplicates.

    Duplicate basenames are warnings, not automatic validation failures, because
    two source lanes may legitimately contain files like ``0001.jpg``. The later
    split/copy logic must still avoid overwriting those names.
    """
    if report.empty:
        return report

    report = report.copy()
    duplicate_base_mask = report.duplicated(
        subset=["base_name"],
        keep=False,
    ) & report["base_name"].ne("")
    for base_name in report.loc[duplicate_base_mask, "base_name"].unique():
        _append_warning(
            report,
            report["base_name"] == base_name,
            "Duplicate base name appears across sources; later split copying must avoid overwrites",
        )

    hash_to_indexes: dict[str, list[int]] = {}
    image_rows = report["image_path"].fillna("").ne("")
    for row_index, image_path_text in report.loc[image_rows, "image_path"].items():
        try:
            file_hash = _hash_file(Path(image_path_text))
        except OSError:
            continue
        hash_to_indexes.setdefault(file_hash, []).append(row_index)

    for row_indexes in hash_to_indexes.values():
        if len(row_indexes) > 1:
            _append_warning(
                report,
                report.index.isin(row_indexes),
                "Exact duplicate image content detected across sources",
            )

    return report


def _validate_sample(
    source: str,
    base_name: str,
    image_path: Path,
    label_path: Path | None,
    allowed_class_ids: set[int],
    duplicate_in_source: bool,
    allow_empty_labels: bool,
) -> dict[str, Any]:
    """Validate one image and its paired YOLO label file."""
    errors: list[str] = []
    warnings: list[str] = []
    image_width: int | None = None
    image_height: int | None = None
    label_stats = LabelStats()

    if duplicate_in_source:
        errors.append("Duplicate image base name inside the same source folder")

    try:
        image_width, image_height = _read_image_size(image_path)
    except ValueError as exc:
        errors.append(str(exc))

    if label_path is None:
        errors.append("Image has no matching label file")
    elif image_width is not None and image_height is not None:
        label_stats, label_errors = _validate_label_file(
            label_path=label_path,
            image_width=image_width,
            image_height=image_height,
            allowed_class_ids=allowed_class_ids,
            allow_empty_labels=allow_empty_labels,
        )
        errors.extend(label_errors)

    return _build_report_row(
        source=source,
        base_name=base_name,
        image_path=image_path,
        label_path=label_path,
        errors=errors,
        warnings=warnings,
        image_width=image_width,
        image_height=image_height,
        label_stats=label_stats,
    )


def _validate_label_file(
    label_path: Path,
    image_width: int,
    image_height: int,
    allowed_class_ids: set[int],
    allow_empty_labels: bool,
) -> tuple[LabelStats, list[str]]:
    """Validate YOLO label rows and return object counts.

    Each non-empty row must follow:

    ``class_id x_center y_center width height``

    Coordinates are normalized, so all four geometry values must be in
    ``[0, 1]`` and the derived pixel-space box must stay inside the image.
    """
    errors: list[str] = []
    label_stats = LabelStats(class_counts={class_id: 0 for class_id in allowed_class_ids})
    seen_rows: dict[tuple[int, float, float, float, float], int] = {}

    try:
        raw_text = label_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return label_stats, [f"Could not read label file: {exc}"]

    if not raw_text:
        if allow_empty_labels:
            return label_stats, []
        return label_stats, ["Label file is empty"]

    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            errors.append(
                f"Line {line_number}: expected 5 values, found {len(values)}"
            )
            continue

        parsed_values = _parse_label_values(values, line_number)
        if isinstance(parsed_values, str):
            errors.append(parsed_values)
            continue

        first_line = seen_rows.get(parsed_values)
        if first_line is not None:
            errors.append(
                f"Line {line_number}: duplicate annotation row; "
                f"matches line {first_line}"
            )
            continue
        seen_rows[parsed_values] = line_number

        class_id, x_center, y_center, box_width, box_height = parsed_values
        if class_id not in allowed_class_ids:
            errors.append(f"Line {line_number}: class_id {class_id} is not allowed")

        geometry_errors = _validate_yolo_geometry(
            line_number=line_number,
            x_center=x_center,
            y_center=y_center,
            box_width=box_width,
            box_height=box_height,
            image_width=image_width,
            image_height=image_height,
        )
        errors.extend(geometry_errors)

        if not geometry_errors and class_id in allowed_class_ids:
            label_stats.num_objects += 1
            label_stats.class_counts[class_id] = (
                label_stats.class_counts.get(class_id, 0) + 1
            )

    return label_stats, errors


def _parse_label_values(
    values: list[str],
    line_number: int,
) -> tuple[int, float, float, float, float] | str:
    """Parse one YOLO label row."""
    try:
        class_id_float = float(values[0])
        coordinates = [float(value) for value in values[1:]]
    except ValueError:
        return f"Line {line_number}: class_id and coordinates must be numeric"

    if not class_id_float.is_integer():
        return f"Line {line_number}: class_id must be an integer"

    class_id = int(class_id_float)
    x_center, y_center, box_width, box_height = coordinates
    return class_id, x_center, y_center, box_width, box_height


def _validate_yolo_geometry(
    line_number: int,
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> list[str]:
    """Validate normalized YOLO coordinates and derived image boundaries."""
    errors: list[str] = []
    coordinate_values = {
        "x_center": x_center,
        "y_center": y_center,
        "width": box_width,
        "height": box_height,
    }

    for coordinate_name, coordinate_value in coordinate_values.items():
        if not 0 <= coordinate_value <= 1:
            errors.append(f"Line {line_number}: {coordinate_name} must be in [0, 1]")

    if box_width <= 0:
        errors.append(f"Line {line_number}: width must be greater than 0")
    if box_height <= 0:
        errors.append(f"Line {line_number}: height must be greater than 0")

    # Normalized values can all be in [0, 1] while the actual box still hangs
    # over an edge, for example x_center=0.01 and width=0.20. The pixel-space
    # check catches that annotation problem before training.
    left = (x_center - box_width / 2) * image_width
    right = (x_center + box_width / 2) * image_width
    top = (y_center - box_height / 2) * image_height
    bottom = (y_center + box_height / 2) * image_height
    tolerance_x = BOUNDARY_TOLERANCE * image_width
    tolerance_y = BOUNDARY_TOLERANCE * image_height

    if left < -tolerance_x or right > image_width + tolerance_x:
        errors.append(f"Line {line_number}: box x-boundaries exceed image")
    if top < -tolerance_y or bottom > image_height + tolerance_y:
        errors.append(f"Line {line_number}: box y-boundaries exceed image")

    return errors


def _collect_image_paths(
    images_dir: Path,
    image_extensions: set[str],
) -> dict[str, list[Path]]:
    """Collect supported image files grouped by base name."""
    image_paths: dict[str, list[Path]] = {}
    for image_path in sorted(images_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            image_paths.setdefault(image_path.stem, []).append(image_path)
    return image_paths


def _collect_unsupported_image_paths(
    images_dir: Path,
    image_extensions: set[str],
) -> list[Path]:
    """Collect non-image files in images/ for explicit validation feedback."""
    return [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() not in image_extensions
    ]


def _collect_label_paths(labels_dir: Path) -> dict[str, Path]:
    """Collect YOLO label files by base name."""
    return {
        label_path.stem: label_path
        for label_path in sorted(labels_dir.iterdir())
        if label_path.is_file() and label_path.suffix.lower() == ".txt"
    }


def _read_image_size(image_path: Path) -> tuple[int, int]:
    """Read image dimensions with Pillow."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("Pillow is required to read image dimensions") from exc

    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception as exc:
        raise ValueError(f"Image is unreadable or corrupted: {exc}") from exc

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    return width, height


def _hash_file(path: Path) -> str:
    """Compute a SHA-256 hash for duplicate-content warnings."""
    digest = sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_warning(
    report: pd.DataFrame,
    row_mask: pd.Series,
    message: str,
) -> None:
    """Append a warning message to matching rows in-place."""
    for row_index in report.index[row_mask]:
        current_warning = str(report.at[row_index, "warnings"] or "")
        if message in current_warning:
            continue
        report.at[row_index, "warnings"] = (
            message if not current_warning else f"{current_warning}; {message}"
        )


def _build_report_row(
    source: str,
    base_name: str,
    image_path: Path | None,
    label_path: Path | None,
    errors: list[str],
    warnings: list[str] | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    label_stats: LabelStats | None = None,
) -> dict[str, Any]:
    """Build a normalized validation report row."""
    label_stats = label_stats or LabelStats()
    warnings = warnings or []
    class_counts = label_stats.class_counts or {}
    return {
        "source": source,
        "base_name": base_name,
        "image_name": image_path.name if image_path else "",
        "label_name": label_path.name if label_path else "",
        "image_path": str(image_path) if image_path else "",
        "label_path": str(label_path) if label_path else "",
        "status": "invalid" if errors else "valid",
        "errors": "; ".join(errors),
        "warnings": "; ".join(warnings),
        "image_width": image_width,
        "image_height": image_height,
        "num_objects": label_stats.num_objects,
        "num_person": class_counts.get(0, 0),
        "num_helmet": class_counts.get(1, 0),
        "num_vest": class_counts.get(2, 0),
        "num_cleaning_coverall": class_counts.get(3, 0),
    }


def _empty_report() -> pd.DataFrame:
    """Return an empty validation report with the expected schema."""
    return pd.DataFrame(columns=VALIDATION_COLUMNS)
