"""Load and validate the small set of PPE v2 pipeline configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


EXPECTED_CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "vest",
    3: "cleaning_coverall",
}
CONFIG_FILENAMES = {
    "classes": "class_names.yaml",
    "dataset": "dataset_config.yaml",
    "augmentation": "augmentation_config.yaml",
    "training": "training_config.yaml",
    "experiments": "experiments.yaml",
}


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest PPE v2 root by its config files.

    Args:
        start: File or directory to search upward from. Defaults to the current
            working directory.

    Returns:
        Resolved path to `ppe-detection/v2_pipeline`.

    Raises:
        FileNotFoundError: If no matching project root is found.
    """
    candidate = Path(start or Path.cwd()).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if (directory / "configs" / "class_names.yaml").is_file() and (
            directory / "src"
        ).is_dir():
            return directory
    raise FileNotFoundError("Could not find the PPE v2 project root")


def load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping with a clear error for invalid top-level data.

    Args:
        path: YAML file to read.

    Returns:
        Parsed top-level mapping. Empty YAML becomes an empty dictionary.

    Raises:
        TypeError: If the YAML root is not a mapping.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return payload


def load_pipeline_config(root: Path | None = None) -> dict[str, Any]:
    """Load all pipeline configs and enforce the fixed data contract.

    Args:
        root: Optional project root. It is discovered when omitted.

    Returns:
        Mapping containing the resolved root, normalized class map, and each
        named configuration mapping.

    Raises:
        ValueError: If classes, source paths, split ratios, or experiments are
            inconsistent with the PPE v2 contract.
    """
    project_root = find_project_root(root)
    config_dir = project_root / "configs"
    loaded = {
        key: load_yaml(config_dir / filename)
        for key, filename in CONFIG_FILENAMES.items()
    }
    loaded["classes"] = normalize_class_names(loaded["classes"])
    _validate_dataset_config(loaded["dataset"])
    _validate_experiments(loaded["experiments"])
    _validate_training_config(loaded["training"])
    loaded["root"] = project_root
    return loaded


def normalize_class_names(payload: dict[str, Any]) -> dict[int, str]:
    """Normalize and validate the single tracked PPE class map.

    Args:
        payload: Parsed `class_names.yaml` payload.

    Returns:
        Integer class IDs mapped to class names.

    Raises:
        ValueError: If `nc` or `names` differs from the fixed four-class map.
    """
    raw_names = payload.get("names", {})
    if not isinstance(raw_names, dict):
        raise ValueError("class_names.yaml must define a names mapping")
    names = {int(class_id): str(name) for class_id, name in raw_names.items()}
    if int(payload.get("nc", -1)) != len(EXPECTED_CLASS_NAMES):
        raise ValueError("class_names.yaml must declare nc: 4")
    if names != EXPECTED_CLASS_NAMES:
        raise ValueError(f"Expected class map {EXPECTED_CLASS_NAMES}, got {names}")
    return names


def resolve_project_path(root: Path, configured_path: str | Path) -> Path:
    """Resolve a configured path against the project root.

    Args:
        root: PPE v2 project root.
        configured_path: Absolute path or project-relative configured path.

    Returns:
        Resolved absolute path.
    """
    path = Path(configured_path)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def get_experiment(config: dict[str, Any], experiment_name: str) -> dict[str, Any]:
    """Return one validated experiment specification.

    Args:
        config: Full mapping returned by :func:`load_pipeline_config`.
        experiment_name: Key from `configs/experiments.yaml`.

    Returns:
        A copy of the experiment specification.

    Raises:
        KeyError: If the experiment name is unknown.
    """
    experiments = config["experiments"]["experiments"]
    if experiment_name not in experiments:
        raise KeyError(f"Unknown experiment: {experiment_name}")
    return dict(experiments[experiment_name])


def resolve_final_experiment(
    config: dict[str, Any],
    ablation_selected_experiment: str | None,
    override: str | None = None,
) -> tuple[str, str]:
    """Resolve the final augmentation experiment and record its decision source.

    Args:
        config: Full mapping returned by :func:`load_pipeline_config`.
        ablation_selected_experiment: Experiment selected by Notebook 05.
        override: Optional explicit final experiment, such as
            ``exp_D_full_pipeline``.

    Returns:
        ``(experiment_name, selection_source)`` where the source is either
        ``ablation`` or ``manual_override``.

    Raises:
        ValueError: If neither the ablation result nor an override is present.
        KeyError: If the resolved experiment is not configured.
    """
    experiment_name = override or ablation_selected_experiment
    if not experiment_name:
        raise ValueError("Final training requires an ablation experiment or explicit override")
    # Validate before test access so a typo cannot silently fall back to D.
    get_experiment(config, str(experiment_name))
    selection_source = "manual_override" if override else "ablation"
    return str(experiment_name), selection_source


def online_augmentation_args(
    augmentation_config: dict[str, Any],
    enabled: bool,
) -> dict[str, float | int]:
    """Return configured online augmentation or neutral values for every key.

    Args:
        augmentation_config: Parsed augmentation config.
        enabled: Whether online augmentation is enabled for the experiment.

    Returns:
        Ultralytics-compatible augmentation arguments.
    """
    configured = dict(augmentation_config.get("online_augmentation", {}))
    if enabled:
        return configured
    return {
        key: 0 if key == "close_mosaic" else 0.0
        for key in configured
    }


def _validate_dataset_config(dataset: dict[str, Any]) -> None:
    """Validate required source paths and factory split ratios."""
    required_sources = {"open_source", "factory_source", "test_source"}
    sources = dataset.get("input", {})
    missing = required_sources - set(sources)
    if missing:
        raise ValueError(f"dataset_config.yaml is missing input paths: {sorted(missing)}")

    split = dataset.get("split", {})
    train_ratio = float(split.get("factory_train", 0))
    val_ratio = float(split.get("factory_val", 0))
    if train_ratio <= 0 or val_ratio <= 0 or abs(train_ratio + val_ratio - 1.0) > 1e-9:
        raise ValueError("factory_train and factory_val must be positive and sum to 1")


def _validate_experiments(experiments_config: dict[str, Any]) -> None:
    """Validate the four expected ablation specifications."""
    experiments = experiments_config.get("experiments", {})
    expected = {
        "exp_A_original_only": ("original", False),
        "exp_B_online_aug": ("original", True),
        "exp_C_offline_aug": ("original_plus_offline", False),
        "exp_D_full_pipeline": ("original_plus_offline", True),
    }
    actual = {
        name: (spec.get("dataset_variant"), bool(spec.get("online_augmentation")))
        for name, spec in experiments.items()
    }
    if actual != expected:
        raise ValueError(f"experiments.yaml must define the fixed A-D matrix: {expected}")


def _validate_training_config(training: dict[str, Any]) -> None:
    """Validate the bounded resolution-calibration search space.

    The calibration notebook intentionally accepts a short explicit profile
    list instead of independently sweeping image size and batch size. This
    keeps comparisons reproducible and prevents accidental combinatorial runs.

    Args:
        training: Parsed ``training_config.yaml`` mapping.

    Raises:
        ValueError: If epochs, profiles, or the ranking contract are invalid.
    """
    if int(training.get("resolution_epochs", 0)) <= 0:
        raise ValueError("resolution_epochs must be a positive integer")

    profiles = training.get("resolution_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("resolution_profiles must be a non-empty list")

    names: list[str] = []
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("Each resolution profile must be a mapping")
        name = str(profile.get("name", "")).strip()
        if not name:
            raise ValueError("Each resolution profile must have a name")
        names.append(name)
        for field in ("imgsz", "batch"):
            value = profile.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"Resolution profile {name!r} must define positive integer {field}"
                )
    if len(names) != len(set(names)):
        raise ValueError("Resolution profile names must be unique")

    expected_priority = [
        "helmet_recall",
        "helmet_map50_95",
        "overall_map50_95",
        "fps",
    ]
    if training.get("resolution_selection_priority") != expected_priority:
        raise ValueError(
            "resolution_selection_priority must preserve the helmet-first contract: "
            f"{expected_priority}"
        )
