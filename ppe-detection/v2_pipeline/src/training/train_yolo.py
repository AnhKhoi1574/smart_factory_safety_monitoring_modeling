"""Minimal Ultralytics training orchestration for candidate, ablation, and final runs."""

from __future__ import annotations

import contextlib
import gc
import io
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from src.training.evaluate_yolo import extract_training_metrics


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


ProgressCallback = Callable[[dict[str, Any]], None]


def build_train_args(
    training_config: dict[str, Any],
    epochs_key: str,
) -> dict[str, Any]:
    """Build shared Ultralytics arguments from the tracked training config.

    Args:
        training_config: Parsed `training_config.yaml`.
        epochs_key: Configured epoch-budget key, such as `candidate_epochs`,
            `ablation_epochs`, `resolution_epochs`, or `final_epochs`.

    Returns:
        Shared non-augmentation training arguments.
    """
    return {
        "epochs": int(training_config[epochs_key]),
        "imgsz": int(training_config["imgsz"]),
        "batch": training_config["batch"],
        "device": training_config["device"],
        "workers": int(training_config["workers"]),
        "cache": training_config.get("cache", False),
        "patience": int(training_config["patience"]),
        "seed": int(training_config["seed"]),
        "amp": bool(training_config["amp"]),
        "plots": False,
        "verbose": False,
        "save_period": -1,
    }


def run_training_specs(
    specs: Iterable[dict[str, Any]],
    models_dir: Path,
    base_train_args: dict[str, Any],
    *,
    dry_run: bool = False,
    continue_on_error: bool = True,
    progress_callback: ProgressCallback | None = None,
    failure_log_lines: int = 30,
) -> pd.DataFrame:
    """Run a small explicit training matrix without embedding experiment policy.

    Each spec requires `run_name`, `model_name`, and `data_yaml`. Optional keys
    are `experiment` and `augmentation_args`.

    Args:
        specs: Explicit training run specifications.
        models_dir: Run-local model artifact directory.
        base_train_args: Shared Ultralytics train arguments.
        dry_run: Validate specs without importing Ultralytics or training.
        continue_on_error: Record a failed row and continue when true.
        progress_callback: Optional receiver for model-start, epoch-end, and
            model-end events. Callback errors never interrupt training.
        failure_log_lines: Maximum captured log lines retained only for a
            failed specification.

    Returns:
        One compact result row per spec.
    """
    output_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_list = [dict(raw_spec) for raw_spec in specs]
    rows = []
    for spec_index, spec in enumerate(spec_list, start=1):
        row = _base_result_row(spec)
        progress_context = {
            "model_index": spec_index,
            "model_total": len(spec_list),
            "run_name": row["run_name"],
            "model_name": row["model_name"],
            "experiment": row["experiment"],
            # Refit runs intentionally disable validation; carrying this flag
            # prevents unavailable metrics from being rendered as real zeroes.
            "validation_enabled": bool(base_train_args.get("val", True)),
        }
        captured_log = io.StringIO()
        _emit_progress(
            progress_callback,
            {
                **progress_context,
                "event": "model_start",
                "epochs": int(base_train_args.get("epochs", 0)),
            },
        )
        try:
            data_yaml = Path(spec["data_yaml"]).resolve()
            if not data_yaml.is_file():
                raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
            if dry_run:
                row.update({"status": "dry_run", "data_yaml": str(data_yaml)})
            else:
                run_dir = _train_one_spec(
                    spec,
                    output_dir,
                    base_train_args,
                    progress_callback=progress_callback,
                    progress_context=progress_context,
                    captured_log=captured_log,
                )
                row.update(
                    {
                        "status": "trained",
                        "run_dir": str(run_dir),
                        "data_yaml": str(data_yaml),
                        **extract_training_metrics(run_dir),
                    }
                )
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "error_message": str(exc),
                    "failure_log_excerpt": _tail_lines(
                        captured_log.getvalue(),
                        failure_log_lines,
                    ),
                }
            )
            _emit_progress(
                progress_callback,
                {
                    **progress_context,
                    "event": "model_end",
                    "status": "failed",
                    "error_message": str(exc),
                },
            )
            if not continue_on_error:
                rows.append(row)
                raise
        else:
            _emit_progress(
                progress_callback,
                {
                    **progress_context,
                    "event": "model_end",
                    "status": row["status"],
                    "recall": row["recall"],
                    "map50": row["map50"],
                    "map50_95": row["map50_95"],
                    "training_time_seconds": row["training_time_seconds"],
                },
            )
        rows.append(row)
    return pd.DataFrame(rows)


def format_progress_event(event: dict[str, Any]) -> str:
    """Format one training event as compact Jupyter-friendly Markdown.

    Args:
        event: Event emitted by :func:`run_training_specs`.

    Returns:
        One or two Markdown lines suitable for an updating display handle.
    """
    run_name = event.get("run_name") or event.get("model_name") or "model"
    model_name = event.get("model_name")
    model_suffix = f" · {model_name}" if model_name and model_name != run_name else ""
    prefix = (
        f"**[{event.get('model_index', '?')}/{event.get('model_total', '?')}] "
        f"{run_name}**{model_suffix}"
    )
    event_name = event.get("event")
    if event_name == "model_start":
        return f"{prefix} — starting, epochs={event.get('epochs', '?')}"
    if event_name == "epoch_end":
        metrics = _formatted_progress_metrics(event)
        details = " | ".join(metrics) if metrics else "metrics unavailable"
        return (
            f"{prefix} — epoch **{int(event.get('epoch', 0)):02d}/"
            f"{int(event.get('epochs', 0)):02d}**  \n{details}"
        )
    if event_name == "model_end":
        status = str(event.get("status", "completed"))
        if status == "failed":
            return f"{prefix} — **failed:** {event.get('error_message', 'unknown error')}"
        if status == "dry_run":
            return f"{prefix} — dry-run contract validated"
        metrics = _formatted_progress_metrics(event)
        suffix = f" | {' | '.join(metrics)}" if metrics else ""
        return f"{prefix} — **completed**{suffix}"
    return f"{prefix} — {event_name or 'status unavailable'}"


def rank_results(
    results: pd.DataFrame,
    priority: Iterable[str],
) -> pd.DataFrame:
    """Rank trained rows lexicographically by the configured validation policy.

    Args:
        results: Training result table.
        priority: Ordered metric names. Model size is minimized; other metrics
            are maximized.

    Returns:
        Stable ranking with failed/dry-run rows after trained rows.
    """
    if results.empty:
        return results.copy()
    ranked = results.copy()
    metrics = [metric for metric in priority if metric in ranked.columns]
    ranked["_trained"] = ranked["status"].eq("trained")
    sort_columns = ["_trained", *metrics]
    ascending = [False, *[metric == "model_size_mb" for metric in metrics]]
    ranked = ranked.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked.drop(columns="_trained")


def retain_best_checkpoint(
    ranked_results: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Copy only the top trained checkpoint to a descriptive retained path.

    Args:
        ranked_results: Output from :func:`rank_results`.
        output_path: Run-local retained checkpoint destination.

    Returns:
        Retained checkpoint path.

    Raises:
        ValueError: If no trained row has a valid best checkpoint.
    """
    return retain_training_checkpoint(
        ranked_results,
        output_path,
        checkpoint_kind="best",
    )


def retain_training_checkpoint(
    training_results: pd.DataFrame,
    output_path: Path,
    *,
    checkpoint_kind: str,
) -> Path:
    """Retain one checkpoint from the first successful training row.

    Args:
        training_results: Compact output from :func:`run_training_specs`.
        output_path: Run-local destination outside the temporary work tree.
        checkpoint_kind: Either `best` for validation-selected workflows or
            `last` for a fixed-epoch refit that disabled validation selection.

    Returns:
        Path to the retained checkpoint copy.

    Raises:
        ValueError: If the checkpoint kind is unknown or no matching checkpoint
            exists in a successful row.
    """
    if checkpoint_kind not in {"best", "last"}:
        raise ValueError(f"Unknown checkpoint kind: {checkpoint_kind}")
    path_column = f"{checkpoint_kind}_weights_path"
    trained = training_results.loc[training_results["status"].eq("trained")]
    for row in trained.to_dict(orient="records"):
        checkpoint = Path(str(row.get(path_column, "")))
        if checkpoint.is_file():
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint, destination)
            return destination
    raise ValueError(
        f"No trained {checkpoint_kind} checkpoint is available to retain"
    )


def clean_training_intermediates(models_dir: Path) -> list[Path]:
    """Remove only this run's `_work` and `_pretrained` directories.

    Args:
        models_dir: Artifact model directory containing generated internals.

    Returns:
        Exact directories removed. Retained checkpoints in `models_dir` remain.

    Raises:
        ValueError: If a cleanup target resolves outside `models_dir`.
    """
    root = Path(models_dir).resolve()
    removed = []
    for name in ("_work", "_pretrained"):
        target = (root / name).resolve()
        if target.parent != root:
            raise ValueError(f"Unsafe training cleanup target: {target}")
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(target)
    return removed


def _train_one_spec(
    spec: dict[str, Any],
    models_dir: Path,
    base_train_args: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None,
    progress_context: dict[str, Any],
    captured_log: io.StringIO,
) -> Path:
    """Run one Ultralytics spec with compact callbacks and captured raw logs."""
    run_name = _safe_name(str(spec["run_name"]))
    work_dir = Path(models_dir) / "_work"
    expected_run_dir = work_dir / run_name
    if expected_run_dir.exists():
        raise FileExistsError(f"Training run already exists: {expected_run_dir}")

    train_args = dict(base_train_args)
    train_args.update(dict(spec.get("augmentation_args", {})))
    train_args.update(
        {
            "data": str(Path(spec["data_yaml"]).resolve()),
            "project": str(work_dir.resolve()),
            "name": run_name,
            "exist_ok": False,
        }
    )
    model = None
    result = None
    try:
        with (
            contextlib.redirect_stdout(captured_log),
            contextlib.redirect_stderr(captured_log),
            _silence_ultralytics_logger(),
        ):
            model = _load_model(str(spec["model_name"]), Path(models_dir) / "_pretrained")
            model.add_callback(
                "on_fit_epoch_end",
                lambda trainer: _report_epoch_progress(
                    trainer,
                    progress_callback,
                    progress_context,
                ),
            )
            result = model.train(**train_args)

        # Ultralytics returns metrics, while the YOLO wrapper retains the
        # trainer that owns the actual run path and both persistent loaders.
        trainer = getattr(model, "trainer", None) or getattr(result, "trainer", None)
        save_dir = getattr(result, "save_dir", None)
        if save_dir is None and trainer is not None:
            save_dir = getattr(trainer, "save_dir", None)
        return Path(save_dir) if save_dir else expected_run_dir
    finally:
        # Windows spawn workers can otherwise survive into the next candidate,
        # exhausting RAM/pagefile even after CUDA tensors have been released.
        _release_training_resources(model, result)


def _load_model(model_name: str, cache_dir: Path):
    """Load a model while containing automatic checkpoint downloads in artifacts."""
    from ultralytics import YOLO

    configured_path = Path(model_name)
    if configured_path.is_file():
        return YOLO(str(configured_path.resolve()))

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cached_checkpoint = cache / configured_path.name
    if cached_checkpoint.is_file():
        return YOLO(str(cached_checkpoint))

    previous_directory = Path.cwd()
    try:
        os.chdir(cache)
        return YOLO(model_name)
    finally:
        os.chdir(previous_directory)


def _release_training_resources(model: Any, result: Any = None) -> None:
    """Release one completed/failed Ultralytics trainer before the next model.

    Args:
        model: Ultralytics YOLO wrapper, or ``None`` when loading failed.
        result: Optional value returned by ``YOLO.train``.

    Side effects:
        Stops persistent train/validation dataloader workers, breaks trainer
        references, runs Python garbage collection, and clears unused CUDA
        allocator blocks. Cleanup errors are contained so they do not mask the
        original training result or exception.
    """
    trainer = getattr(model, "trainer", None) or getattr(result, "trainer", None)
    if trainer is not None:
        validator = getattr(trainer, "validator", None)
        if validator is not None:
            _safe_setattr(validator, "dataloader", None)

        for loader_name in ("train_loader", "test_loader"):
            loader = getattr(trainer, loader_name, None)
            _shutdown_dataloader(loader)
            _safe_setattr(trainer, loader_name, None)

        # These objects retain datasets, tensors, and optimizer state after the
        # checkpoint and metrics have already been written to the run folder.
        for attribute in ("optimizer", "validator", "model"):
            _safe_setattr(trainer, attribute, None)

    if model is not None:
        for attribute in ("trainer", "model", "ckpt", "metrics"):
            _safe_setattr(model, attribute, None)

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Resource cleanup must not replace a useful training exception with a
        # secondary optional-runtime error.
        pass


def _shutdown_dataloader(loader: Any) -> None:
    """Shut down a PyTorch/Ultralytics dataloader iterator when present."""
    if loader is None:
        return
    iterators = {
        id(iterator): iterator
        for iterator in (
            getattr(loader, "iterator", None),
            getattr(loader, "_iterator", None),
        )
        if iterator is not None
    }
    for iterator in iterators.values():
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                # Ultralytics' own InfiniteDataLoader destructor also treats
                # worker shutdown as best-effort during object destruction.
                pass


def _safe_setattr(target: Any, attribute: str, value: Any) -> None:
    """Set a cleanup attribute without allowing version drift to mask errors."""
    try:
        setattr(target, attribute, value)
    except Exception:
        pass


def _base_result_row(spec: dict[str, Any]) -> dict[str, Any]:
    """Create one predictable result row before training starts."""
    return {
        "run_name": str(spec.get("run_name", "")),
        "model_name": str(spec.get("model_name", "")),
        "experiment": str(spec.get("experiment", "")),
        "data_yaml": str(spec.get("data_yaml", "")),
        "online_augmentation": bool(spec.get("online_augmentation", False)),
        "status": "not_started",
        "run_dir": "",
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "fps": None,
        "avg_inference_ms": None,
        "model_size_mb": None,
        "training_time_seconds": None,
        "best_weights_path": "",
        "last_weights_path": "",
        "error_message": "",
        "failure_log_excerpt": "",
    }


def _safe_name(value: str) -> str:
    """Return a simple filesystem-safe Ultralytics run name."""
    normalized = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return normalized.strip("-") or "training-run"


def _report_epoch_progress(
    trainer: Any,
    callback: ProgressCallback | None,
    context: dict[str, Any],
) -> None:
    """Emit train losses and validation metrics after one completed epoch."""
    losses = trainer.label_loss_items(trainer.tloss) if trainer.tloss is not None else {}
    metrics = dict(trainer.metrics or {})
    _emit_progress(
        callback,
        {
            **context,
            "event": "epoch_end",
            "epoch": int(trainer.epoch) + 1,
            "epochs": int(trainer.epochs),
            "box_loss": _metric_value(losses, ("box_loss",)),
            "cls_loss": _metric_value(losses, ("cls_loss",)),
            "dfl_loss": _metric_value(losses, ("dfl_loss",)),
            "precision": _metric_value(metrics, ("precision",)),
            "recall": _metric_value(metrics, ("recall",)),
            "map50": _metric_value(metrics, ("map50",), exclude=("95",)),
            "map50_95": _metric_value(metrics, ("map50-95", "map50_95")),
            "elapsed_seconds": max(
                0.0,
                time.time() - float(getattr(trainer, "train_time_start", time.time())),
            ),
        },
    )


def _emit_progress(
    callback: ProgressCallback | None,
    event: dict[str, Any],
) -> None:
    """Send a copied progress event without allowing UI errors to stop training."""
    if callback is None:
        return
    try:
        callback(dict(event))
    except Exception:
        # Notebook display is observational; training correctness must not
        # depend on a frontend display handle remaining available.
        return


def _metric_value(
    values: dict[str, Any],
    patterns: Iterable[str],
    *,
    exclude: Iterable[str] = (),
) -> float | None:
    """Find a finite scalar by tolerant metric-key matching."""
    normalized_patterns = tuple(pattern.lower() for pattern in patterns)
    normalized_exclude = tuple(pattern.lower() for pattern in exclude)
    for key, value in values.items():
        normalized_key = str(key).lower()
        if any(pattern in normalized_key for pattern in normalized_patterns) and not any(
            blocked in normalized_key for blocked in normalized_exclude
        ):
            try:
                converted = float(value)
            except (TypeError, ValueError):
                return None
            return converted if pd.notna(converted) else None
    return None


def _formatted_progress_metrics(event: dict[str, Any]) -> list[str]:
    """Return bounded metric labels for one progress display event."""
    train_labels = (
        ("box", "box_loss"),
        ("cls", "cls_loss"),
        ("dfl", "dfl_loss"),
    )
    validation_labels = (
        ("P", "precision"),
        ("R", "recall"),
        ("mAP50", "map50"),
        ("mAP50-95", "map50_95"),
    )
    labels = (
        (*train_labels, *validation_labels)
        if event.get("validation_enabled", True)
        else train_labels
    )
    formatted = []
    for label, key in labels:
        value = event.get(key)
        if value is not None and pd.notna(value):
            formatted.append(f"{label}={float(value):.4f}")
    duration = event.get("elapsed_seconds", event.get("training_time_seconds"))
    if duration is not None and pd.notna(duration):
        formatted.append(f"elapsed={_format_duration(float(duration))}")
    return formatted


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds as a compact hours/minutes/seconds value."""
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return (
        f"{hours:d}:{minutes:02d}:{seconds_part:02d}"
        if hours
        else f"{minutes:02d}:{seconds_part:02d}"
    )


def _tail_lines(text: str, max_lines: int) -> str:
    """Return only the final non-empty lines from a captured failure log."""
    if max_lines <= 0:
        return ""
    lines = [line.rstrip() for line in str(text).splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


@contextlib.contextmanager
def _silence_ultralytics_logger():
    """Temporarily disable Ultralytics' pre-bound logger and restore it safely."""
    from ultralytics.utils import LOGGER

    was_disabled = LOGGER.disabled
    LOGGER.disabled = True
    try:
        yield
    finally:
        LOGGER.disabled = was_disabled
