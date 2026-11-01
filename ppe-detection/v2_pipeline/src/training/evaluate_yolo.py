"""Compact metric extraction, latency, and final-evaluation helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.dataset.manifests import read_manifest


def extract_training_metrics(run_dir: Path) -> dict[str, Any]:
    """Read comparable metrics from one Ultralytics `results.csv`.

    Args:
        run_dir: Native Ultralytics training run directory.

    Returns:
        Validation metrics and checkpoint paths. Missing values remain `None`.
    """
    run_path = Path(run_dir)
    best_weights = run_path / "weights" / "best.pt"
    last_weights = run_path / "weights" / "last.pt"
    metrics: dict[str, Any] = {
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "training_time_seconds": None,
        "best_weights_path": str(best_weights) if best_weights.is_file() else "",
        "last_weights_path": str(last_weights) if last_weights.is_file() else "",
        "model_size_mb": round(best_weights.stat().st_size / 1_048_576, 3)
        if best_weights.is_file()
        else None,
    }
    results_path = run_path / "results.csv"
    if not results_path.is_file():
        return metrics
    results = pd.read_csv(results_path)
    if results.empty:
        return metrics
    row = results.iloc[-1]
    metrics.update(
        {
            "precision": _row_value(row, ("metrics/precision", "precision")),
            "recall": _row_value(row, ("metrics/recall", "recall")),
            "map50": _row_value(row, ("metrics/mAP50(B)", "map50"), exclude=("95",)),
            "map50_95": _row_value(
                row,
                ("metrics/mAP50-95(B)", "map50-95", "map50_95"),
            ),
            "training_time_seconds": _row_value(row, ("time",)),
        }
    )
    return metrics


def extract_best_epoch(run_dir: Path) -> dict[str, Any]:
    """Select the first epoch with maximum validation mAP50-95.

    Ultralytics detection fitness currently uses mAP50-95, so this reproduces
    the epoch decision behind `best.pt` while returning a compact auditable row
    for a later train-plus-validation refit.

    Args:
        run_dir: Completed calibration run containing `results.csv`.

    Returns:
        Best one-based epoch and its validation precision, recall, mAP50, and
        mAP50-95 values.

    Raises:
        FileNotFoundError: If the native results CSV is missing.
        ValueError: If no finite mAP50-95 value can select an epoch.
    """
    results_path = Path(run_dir) / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Training results do not exist: {results_path}")
    results = pd.read_csv(results_path)
    if results.empty:
        raise ValueError(f"Training results are empty: {results_path}")

    metric_column = next(
        (
            column
            for column in results.columns
            if "map50-95" in str(column).strip().lower()
            or "map50_95" in str(column).strip().lower()
        ),
        None,
    )
    if metric_column is None:
        raise ValueError("Calibration results contain no mAP50-95 column")
    metric_values = pd.to_numeric(results[metric_column], errors="coerce")
    if not metric_values.notna().any():
        raise ValueError("Calibration results contain no finite mAP50-95 values")

    # `idxmax` returns the first maximum, matching Ultralytics' strict
    # greater-than update when equal fitness values occur later.
    best_index = metric_values.idxmax()
    row = results.loc[best_index]
    epoch_value = _row_value(row, ("epoch",))
    best_epoch = int(epoch_value) if epoch_value is not None else int(best_index) + 1
    epoch_column = next(
        (
            column
            for column in results.columns
            if str(column).strip().lower() == "epoch"
        ),
        None,
    )
    epoch_series = (
        pd.to_numeric(results[epoch_column], errors="coerce")
        if epoch_column is not None
        else pd.Series(dtype=float)
    )
    if epoch_series.notna().any() and float(epoch_series.min()) == 0:
        # Older result formats may store every epoch as a zero-based index.
        best_epoch += 1
    return {
        "best_epoch": best_epoch,
        "selection_metric": "map50_95",
        "precision": _row_value(row, ("metrics/precision", "precision")),
        "recall": _row_value(row, ("metrics/recall", "recall")),
        "map50": _row_value(row, ("metrics/mAP50(B)", "map50"), exclude=("95",)),
        "map50_95": _safe_float(metric_values.loc[best_index]),
    }


def benchmark_model_latency(
    weights_path: Path,
    image_manifest: Path,
    *,
    imgsz: int,
    device: str | int,
    max_images: int = 20,
    warmup_images: int = 2,
) -> dict[str, Any]:
    """Benchmark one model on a bounded validation-image sample.

    Args:
        weights_path: Trained checkpoint.
        image_manifest: Validation image manifest; never a test manifest during
            model selection.
        imgsz: Ultralytics inference image size.
        device: Ultralytics device selector.
        max_images: Maximum timed images.
        warmup_images: Number of untimed warmup predictions.

    Returns:
        Average latency, FPS, image count, and status.
    """
    from ultralytics import YOLO

    images = read_manifest(image_manifest)[: max(1, int(max_images))]
    if not images:
        raise ValueError(f"Latency manifest is empty: {image_manifest}")
    model = YOLO(str(Path(weights_path)))
    for image_path in images[:warmup_images]:
        model.predict(str(image_path), imgsz=imgsz, device=device, verbose=False)

    timings = []
    for image_path in images:
        started = time.perf_counter()
        model.predict(str(image_path), imgsz=imgsz, device=device, verbose=False)
        timings.append(time.perf_counter() - started)
    average_seconds = sum(timings) / len(timings)
    return {
        "latency_status": "benchmarked",
        "benchmark_images": len(timings),
        "avg_inference_ms": average_seconds * 1_000,
        "fps": 1.0 / average_seconds if average_seconds > 0 else None,
    }


def evaluate_model(
    weights_path: Path,
    dataset_yaml: Path,
    *,
    split: str,
    imgsz: int,
    device: str | int,
) -> dict[str, Any]:
    """Evaluate a locked model and return compact overall/class metrics.

    Args:
        weights_path: Trained checkpoint.
        dataset_yaml: Run-local dataset config. Final evaluation includes test.
        split: Ultralytics split name, normally `test` for the final run.
        imgsz: Evaluation image size.
        device: Ultralytics device selector.

    Returns:
        Mapping with one overall metric dictionary, a class metric DataFrame,
        and an optional confusion matrix array.
    """
    from ultralytics import YOLO

    model = YOLO(str(Path(weights_path)))
    result = model.val(
        data=str(Path(dataset_yaml)),
        split=split,
        imgsz=imgsz,
        device=device,
        plots=False,
        verbose=False,
    )
    overall = _metrics_from_result(result)
    class_metrics = _class_metrics_from_result(result)
    confusion = getattr(getattr(result, "confusion_matrix", None), "matrix", None)
    return {
        "overall": overall,
        "class_metrics": class_metrics,
        "confusion_matrix": np.asarray(confusion) if confusion is not None else None,
    }


def save_confusion_matrix(
    matrix: np.ndarray,
    labels: Iterable[str],
    output_path: Path,
) -> Path:
    """Save one labeled confusion-matrix figure without extra plot artifacts."""
    import matplotlib.pyplot as plt

    values = np.asarray(matrix)
    axis_labels = list(labels)
    if values.shape[0] == len(axis_labels) + 1:
        axis_labels.append("background")
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values, cmap="Blues")
    axis.set_title("Final untouched-test confusion matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(len(axis_labels)), axis_labels, rotation=30, ha="right")
    axis.set_yticks(range(len(axis_labels)), axis_labels)
    figure.colorbar(image, ax=axis, label="Detections")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_prediction_grid(
    weights_path: Path,
    image_manifest: Path,
    output_path: Path,
    *,
    imgsz: int,
    device: str | int,
    conf: float,
    max_images: int = 6,
) -> Path:
    """Render a bounded final-test prediction sample into one contact sheet."""
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    image_paths = read_manifest(image_manifest)[: max(1, int(max_images))]
    model = YOLO(str(Path(weights_path)))
    predictions = model.predict(
        [str(path) for path in image_paths],
        imgsz=imgsz,
        device=device,
        conf=conf,
        verbose=False,
    )
    columns = 3
    rows = int(np.ceil(len(predictions) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, rows * 3.5), squeeze=False)
    for axis, prediction, source_path in zip(axes.flat, predictions, image_paths):
        rendered_bgr = prediction.plot()
        axis.imshow(rendered_bgr[..., ::-1])
        axis.set_title(source_path.name)
        axis.axis("off")
    for axis in list(axes.flat)[len(predictions):]:
        axis.axis("off")
    figure.suptitle(f"Final untouched-test prediction sample (images={len(predictions)})", y=1.01)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def _metrics_from_result(result: Any) -> dict[str, Any]:
    """Extract common aggregate metrics from an Ultralytics result object."""
    box = getattr(result, "box", None)
    results_dict = getattr(result, "results_dict", {}) or {}
    return {
        "precision": _first_available(
            _safe_float(getattr(box, "mp", None)),
            _dict_value(results_dict, ("precision",)),
        ),
        "recall": _first_available(
            _safe_float(getattr(box, "mr", None)),
            _dict_value(results_dict, ("recall",)),
        ),
        "map50": _first_available(
            _safe_float(getattr(box, "map50", None)),
            _dict_value(results_dict, ("map50",), exclude=("95",)),
        ),
        "map50_95": _first_available(
            _safe_float(getattr(box, "map", None)),
            _dict_value(results_dict, ("map50-95", "map50_95")),
        ),
    }


def _class_metrics_from_result(result: Any) -> pd.DataFrame:
    """Extract one class-level row per configured class when available."""
    names = getattr(result, "names", {}) or {}
    box = getattr(result, "box", None)
    rows = []
    for class_id, class_name in sorted(names.items()):
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": str(class_name),
                "precision": _sequence_value(getattr(box, "p", None), int(class_id)),
                "recall": _sequence_value(getattr(box, "r", None), int(class_id)),
                "map50": _sequence_value(getattr(box, "ap50", None), int(class_id)),
                "map50_95": _sequence_value(getattr(box, "maps", None), int(class_id)),
            }
        )
    return pd.DataFrame(rows)


def _row_value(
    row: pd.Series,
    patterns: Iterable[str],
    *,
    exclude: Iterable[str] = (),
) -> float | None:
    """Find a numeric metric in a result row using tolerant column patterns."""
    normalized_patterns = tuple(pattern.lower() for pattern in patterns)
    normalized_exclude = tuple(pattern.lower() for pattern in exclude)
    for column in row.index:
        normalized = str(column).strip().lower()
        if any(pattern.lower() == normalized for pattern in normalized_patterns):
            return _safe_float(row[column])
    for column in row.index:
        normalized = str(column).strip().lower()
        if any(pattern in normalized for pattern in normalized_patterns) and not any(
            blocked in normalized for blocked in normalized_exclude
        ):
            return _safe_float(row[column])
    return None


def _dict_value(
    values: dict[str, Any],
    patterns: Iterable[str],
    *,
    exclude: Iterable[str] = (),
) -> float | None:
    """Find a numeric metric in a result dictionary by tolerant key matching."""
    row = pd.Series(values, dtype=object)
    return _row_value(row, patterns, exclude=exclude)


def _safe_float(value: Any) -> float | None:
    """Convert finite scalar-like values to float and reject unavailable data."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _sequence_value(values: Any, index: int) -> float | None:
    """Read one numeric index from a tensor/list-like metric sequence."""
    if values is None:
        return None
    try:
        return _safe_float(values[index])
    except (IndexError, KeyError, TypeError):
        return None


def _first_available(*values: float | None) -> float | None:
    """Return the first metric that is not `None`, preserving valid zeroes."""
    return next((value for value in values if value is not None), None)
