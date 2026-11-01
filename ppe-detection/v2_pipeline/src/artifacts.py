"""Create small, run-scoped artifact directories for PPE experiments."""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Resolved directories belonging to one immutable experiment run."""

    root: Path
    manifests: Path
    figures: Path
    csv: Path
    models: Path

    @property
    def run_id(self) -> str:
        """Return the directory name used as the run identifier."""
        return self.root.name


def create_run(
    project_root: Path,
    stage: str,
    descriptor: str,
    seed: int,
    *,
    timestamp: datetime | None = None,
    git_sha: str | None = None,
) -> RunPaths:
    """Create one unique artifact directory and its standard subdirectories.

    Args:
        project_root: PPE v2 project root.
        stage: Short workflow stage, such as `data-audit` or `ablation`.
        descriptor: Human-readable model or dataset descriptor.
        seed: Experiment random seed.
        timestamp: Optional timestamp for deterministic tests.
        git_sha: Optional Git revision for deterministic tests.

    Returns:
        Paths for the newly created run.

    Raises:
        FileExistsError: If the generated run ID already exists.
    """
    moment = timestamp or datetime.now(timezone.utc)
    timestamp_text = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    revision = _slug(git_sha or current_git_sha(project_root) or "nogit")[:8]
    run_id = (
        f"{timestamp_text}_{_slug(stage)}_{_slug(descriptor)}_"
        f"s{int(seed)}_{revision}"
    )
    run_root = Path(project_root) / "artifacts" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    paths = RunPaths(
        root=run_root,
        manifests=run_root / "manifests",
        figures=run_root / "figures",
        csv=run_root / "csv",
        models=run_root / "models",
    )
    for directory in (paths.manifests, paths.figures, paths.csv, paths.models):
        directory.mkdir()
    return paths


def open_run(run_root: Path) -> RunPaths:
    """Resolve an existing run without creating or modifying directories.

    Args:
        run_root: Existing artifact run directory.

    Returns:
        Standard run paths.

    Raises:
        FileNotFoundError: If the run directory does not exist.
    """
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Artifact run does not exist: {root}")
    return RunPaths(
        root=root,
        manifests=root / "manifests",
        figures=root / "figures",
        csv=root / "csv",
        models=root / "models",
    )


def write_run_metadata(run: RunPaths, payload: dict[str, Any]) -> Path:
    """Write or replace the concise metadata record for one run.

    Args:
        run: Target run paths.
        payload: YAML-serializable experiment metadata.

    Returns:
        Path to `run.yaml`.
    """
    metadata_path = run.root / "run.yaml"
    normalized = {"run_id": run.run_id, **payload}
    _write_yaml(metadata_path, normalized)
    return metadata_path


def update_run_metadata(run: RunPaths, updates: dict[str, Any]) -> Path:
    """Merge selected values into an existing run metadata record.

    Args:
        run: Target run paths.
        updates: Top-level values to add or replace.

    Returns:
        Path to the updated `run.yaml`.
    """
    metadata_path = run.root / "run.yaml"
    payload = read_run_metadata(run) if metadata_path.exists() else {}
    payload.update(updates)
    return write_run_metadata(run, payload)


def read_run_metadata(run: RunPaths | Path) -> dict[str, Any]:
    """Read one run metadata mapping.

    Args:
        run: Run paths or artifact run directory.

    Returns:
        Parsed `run.yaml` mapping.

    Raises:
        FileNotFoundError: If `run.yaml` is absent.
        TypeError: If the YAML root is not a mapping.
    """
    root = run.root if isinstance(run, RunPaths) else Path(run)
    metadata_path = root / "run.yaml"
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {metadata_path}")
    return payload


def find_latest_run(
    project_root: Path,
    stage: str,
    required_relative_path: str | Path | None = None,
) -> RunPaths:
    """Find the newest run for a stage, optionally requiring one output file.

    Args:
        project_root: PPE v2 project root.
        stage: Stage token used in the run ID.
        required_relative_path: File that must exist inside a candidate run.

    Returns:
        Paths for the newest matching run.

    Raises:
        FileNotFoundError: If no complete matching run exists.
    """
    artifacts_dir = Path(project_root) / "artifacts"
    stage_token = f"_{_slug(stage)}_"
    candidates = (
        sorted(
            (
                path
                for path in artifacts_dir.iterdir()
                if path.is_dir() and stage_token in path.name
            ),
            reverse=True,
        )
        if artifacts_dir.exists()
        else []
    )
    for candidate in candidates:
        if required_relative_path is None or (candidate / required_relative_path).is_file():
            return open_run(candidate)
    raise FileNotFoundError(f"No completed artifact run found for stage '{stage}'")


def runtime_metadata(project_root: Path) -> dict[str, Any]:
    """Collect lightweight runtime details for reproducibility metadata.

    Args:
        project_root: Repository path used to resolve the Git revision.

    Returns:
        Python, platform, package, and Git metadata. Missing optional packages
        are recorded as `not-installed`.
    """
    packages = {}
    for package_name in ("ultralytics", "torch", "pandas", "Pillow", "PyYAML"):
        try:
            packages[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            packages[package_name] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": current_git_sha(project_root) or "unknown",
        "packages": packages,
        "accelerator": _accelerator_metadata(),
    }


def current_git_sha(project_root: Path) -> str:
    """Return the current short Git revision or an empty string outside Git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _accelerator_metadata() -> dict[str, Any]:
    """Report CUDA and GPU details without making PyTorch a module import dependency."""
    try:
        import torch
    except Exception as exc:  # A broken optional CUDA install must not block data audit.
        return {
            "cuda_available": False,
            "cuda_version": None,
            "gpu_devices": [],
            "inspection_error": str(exc),
        }

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    return {
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_devices": [torch.cuda.get_device_name(index) for index in range(device_count)],
    }


def _slug(value: str) -> str:
    """Convert text into a non-empty Windows-safe run ID component."""
    slug = re.sub(r"[^A-Za-z0-9-]+", "-", str(value).strip()).strip("-").lower()
    return slug or "run"


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one small YAML mapping."""
    path = Path(path)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    temporary_path.replace(path)
