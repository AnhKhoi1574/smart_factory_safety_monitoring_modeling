"""Create descriptive experiment folders and save the approved final artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import matplotlib.figure
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

EXPERIMENT_ID_PATTERN = re.compile(
    r"^(?P<date>\d{8})_[a-z0-9]+(?:_[a-z0-9]+)*_v\d{2}$"
)


def validate_experiment_id(experiment_id: str) -> None:
    """Validate a descriptive, versioned experiment identifier.

    Args:
        experiment_id: Identifier following
            ``YYYYMMDD_model_family_short_purpose_vNN``.

    Raises:
        ValueError: If the format is invalid or the date component is not a real
            calendar date.
    """
    match = EXPERIMENT_ID_PATTERN.fullmatch(experiment_id)
    if match is None:
        raise ValueError(
            "Experiment ID must follow "
            "YYYYMMDD_model_family_short_purpose_vNN using lowercase letters, "
            "digits, and underscores."
        )
    try:
        datetime.strptime(match.group("date"), "%Y%m%d")
    except ValueError as exc:
        raise ValueError("Experiment ID begins with an invalid date.") from exc


def create_experiment_paths(
    artifact_root: Path,
    experiment_id: str,
) -> dict[str, Path]:
    """Create one new experiment directory with exactly three subfolders.

    Args:
        artifact_root: Parent directory that contains all experiment runs.
        experiment_id: Valid descriptive experiment identifier.

    Returns:
        Paths named ``root``, ``figures``, ``csv``, and ``model``.

    Raises:
        FileExistsError: If the experiment directory already exists. Existing
            experiments are never silently overwritten.
        ValueError: If the experiment identifier is invalid.

    Side Effects:
        Creates the experiment directory and its three required subfolders.
    """
    validate_experiment_id(experiment_id)
    experiment_root = artifact_root / experiment_id
    if experiment_root.exists():
        raise FileExistsError(
            f"Experiment already exists: {experiment_root}. Use a new versioned ID."
        )

    paths = {
        "root": experiment_root,
        "figures": experiment_root / "figures",
        "csv": experiment_root / "csv",
        "model": experiment_root / "model",
    }
    for key in ("figures", "csv", "model"):
        paths[key].mkdir(parents=True, exist_ok=False)
    return paths


def save_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    """Save one approved tabular artifact without an implicit index.

    Args:
        frame: DataFrame to serialize.
        path: Descriptive CSV destination inside the current experiment.

    Returns:
        Written path.

    Side Effects:
        Writes a CSV file and creates its immediate parent when necessary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def save_figure(
    figure: matplotlib.figure.Figure,
    path: Path,
    *,
    dpi: int = 160,
) -> Path:
    """Save one fully labeled figure with a tight bounding box.

    Args:
        figure: Matplotlib figure already labeled and annotated by the notebook.
        path: Descriptive PNG destination inside the current experiment.
        dpi: Output resolution.

    Returns:
        Written path.

    Side Effects:
        Writes a figure file and creates its immediate parent when necessary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _json_ready(value: Any) -> Any:
    """Convert common scientific Python values to JSON-safe representations."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def relevant_library_versions() -> dict[str, str]:
    """Return versions needed to reproduce or load the selected model.

    Returns:
        Installed versions for the core and optional modeling libraries. Missing
        optional packages are reported as ``not-installed``.
    """
    distributions = [
        "numpy",
        "scikit-learn",
        "joblib",
        "xgboost",
        "lightgbm",
    ]
    versions = {}
    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def save_model_bundle(
    model: ClassifierMixin,
    metadata_payload: Mapping[str, Any],
    model_dir: Path,
) -> list[Path]:
    """Save the selected estimator and its usage metadata.

    Args:
        model: Final fitted behavior classifier.
        metadata_payload: Feature order, class mapping, selection details, and
            metrics required to use and interpret the model safely.
        model_dir: Current experiment's ``model`` subfolder.

    Returns:
        Paths to the serialized model and metadata JSON.

    Side Effects:
        Writes exactly two approved files to ``model_dir``.
    """
    model_path = model_dir / "selected_behavior_classifier.joblib"
    metadata_path = model_dir / "selected_behavior_classifier_metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps(_json_ready(metadata_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return [model_path, metadata_path]


def format_saved_artifacts(paths: Sequence[Path], experiment_root: Path) -> str:
    """Format the explicit saved-file allowlist for concise notebook output.

    Args:
        paths: Exact artifact paths written during the current run.
        experiment_root: Root used to display relative paths.

    Returns:
        Newline-separated relative path list. No directory scan is performed.
    """
    return "\n".join(f"- {path.relative_to(experiment_root)}" for path in paths)
