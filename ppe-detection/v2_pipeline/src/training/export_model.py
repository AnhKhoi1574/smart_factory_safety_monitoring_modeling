"""Export only requested final deployment formats into the final run artifact."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def export_final_model(
    weights_path: Path,
    export_formats: Iterable[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Export a final checkpoint and retain one descriptive file per format.

    Args:
        weights_path: Locked final checkpoint.
        export_formats: Ultralytics export format names, such as `onnx`.
        output_dir: Run-local `models/` directory.

    Returns:
        One status row per requested format. Machine-dependent failures are
        recorded instead of aborting other formats.
    """
    from ultralytics import YOLO

    checkpoint = Path(weights_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Final checkpoint does not exist: {checkpoint}")
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(checkpoint))
    rows: list[dict[str, Any]] = []
    for export_format in export_formats:
        row = {
            "format": str(export_format),
            "status": "not_started",
            "output_path": "",
            "error_message": "",
        }
        try:
            exported = model.export(format=str(export_format))
            source = _export_path(exported)
            target = destination_dir / f"final_model{source.suffix}"
            if source.resolve() != target.resolve():
                if target.exists():
                    raise FileExistsError(f"Export target already exists: {target}")
                shutil.move(str(source), str(target))
            row.update({"status": "exported", "output_path": str(target)})
        except Exception as exc:
            row.update({"status": "failed", "error_message": str(exc)})
        rows.append(row)
    return pd.DataFrame(rows)


def _export_path(exported: Any) -> Path:
    """Normalize common Ultralytics export return values to one file path."""
    if isinstance(exported, (str, Path)):
        return Path(exported)
    if isinstance(exported, list) and exported:
        return Path(exported[0])
    raise ValueError(f"Could not resolve exported model path from {exported!r}")
