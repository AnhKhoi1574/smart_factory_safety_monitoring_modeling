"""Focused plotting helpers used by the PPE experiment notebooks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image


def plot_class_distribution(distribution: pd.DataFrame, output_path: Path):
    """Plot grouped source/class object counts and save one labeled figure."""
    plt = _pyplot()
    pivot = distribution.pivot(index="class_name", columns="source", values="object_count")
    figure, axis = plt.subplots(figsize=(9, 4.8))
    pivot.plot.bar(ax=axis)
    axis.set_title(f"PPE object distribution by source (n={int(distribution.object_count.sum()):,})")
    axis.set_xlabel("Class")
    axis.set_ylabel("Annotated objects")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(title="Source")
    _save(figure, output_path)
    return figure


def plot_split_balance(summary: pd.DataFrame, output_path: Path):
    """Plot class counts for the frozen train/validation/test manifests."""
    plt = _pyplot()
    class_columns = [
        column
        for column in summary
        if column.startswith("num_") and column != "num_images"
    ]
    plot_frame = summary.set_index("split")[class_columns]
    plot_frame.columns = [column.removeprefix("num_") for column in plot_frame.columns]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    plot_frame.plot.bar(ax=axis)
    axis.set_title(f"Frozen split class balance (images={int(summary.num_images.sum()):,})")
    axis.set_xlabel("Split")
    axis.set_ylabel("Annotated objects")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Class")
    _save(figure, output_path)
    return figure


def plot_metric_comparison(
    results: pd.DataFrame,
    label_column: str,
    metrics: Iterable[str],
    output_path: Path,
    title: str,
):
    """Plot a small set of comparable validation metrics."""
    metric_columns = [metric for metric in metrics if metric in results]
    if not metric_columns:
        raise ValueError("No requested metric columns are available")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9, 4.8))
    results.set_index(label_column)[metric_columns].plot.bar(ax=axis)
    axis.set_title(f"{title} (runs={len(results)})")
    axis.set_xlabel(label_column.replace("_", " ").title())
    axis.set_ylabel("Validation metric")
    axis.set_ylim(bottom=0)
    axis.tick_params(axis="x", rotation=20)
    axis.legend(title="Metric")
    _save(figure, output_path)
    return figure


def plot_augmentation_contact_sheet(
    report: pd.DataFrame,
    output_path: Path,
    *,
    samples_per_transform: int = 2,
):
    """Save paired original/augmented examples with transform captions."""
    selected = (
        report.loc[report["status"].eq("generated")]
        .groupby("augmentation_type", sort=True)
        .head(samples_per_transform)
        .reset_index(drop=True)
    )
    if selected.empty:
        raise ValueError("No generated augmentation examples are available")

    plt = _pyplot()
    rows = len(selected)
    # Give each image column enough horizontal room for a readable caption.
    figure, axes = plt.subplots(
        rows,
        2,
        figsize=(13, max(3.4, rows * 3.1)),
        squeeze=False,
    )
    for row_index, row in enumerate(selected.itertuples(index=False)):
        with Image.open(row.original_image_path) as image:
            original = image.convert("RGB")
        with Image.open(row.augmented_image_path) as image:
            augmented = image.convert("RGB")
        axes[row_index, 0].imshow(original)
        axes[row_index, 0].set_title(
            f"Original\n{_compact_filename(row.original_image_path)}",
            fontsize=9,
            pad=7,
        )
        axes[row_index, 1].imshow(augmented)
        axes[row_index, 1].set_title(
            f"{str(row.augmentation_type).replace('_', ' ').title()}\n"
            f"{_compact_filename(row.augmented_image_path)}",
            fontsize=9,
            pad=7,
        )
        for axis in axes[row_index]:
            axis.axis("off")
    figure.suptitle(f"Offline augmentation visual QA (pairs={rows})", y=1.01)
    figure.subplots_adjust(wspace=0.24, hspace=0.38)
    _save(figure, output_path)
    return figure


def plot_image_grid(
    image_paths: Iterable[Path],
    output_path: Path,
    *,
    title: str,
    columns: int = 3,
):
    """Combine a small set of already annotated images into one figure."""
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("No images were supplied for the grid")
    plt = _pyplot()
    rows = math.ceil(len(paths) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 3.2), squeeze=False)
    for axis, image_path in zip(axes.flat, paths):
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(image_path.name)
        axis.axis("off")
    for axis in list(axes.flat)[len(paths):]:
        axis.axis("off")
    figure.suptitle(f"{title} (images={len(paths)})", y=1.01)
    _save(figure, output_path)
    return figure


def _pyplot():
    """Import Matplotlib lazily so non-visual tests stay lightweight."""
    import matplotlib.pyplot as plt

    return plt


def _compact_filename(path: str | Path, max_chars: int = 38) -> str:
    """Shorten a long filename in the middle while preserving its extension.

    Args:
        path: Image path shown in a figure title.
        max_chars: Maximum displayed filename length, including the extension.

    Returns:
        Original filename when short enough, otherwise a head/tail abbreviation.
    """
    name = Path(path).name
    if len(name) <= max_chars:
        return name

    suffix = Path(name).suffix
    stem = Path(name).stem
    available = max(8, max_chars - len(suffix) - 3)
    head_length = (available + 1) // 2
    tail_length = available // 2
    return f"{stem[:head_length]}...{stem[-tail_length:]}{suffix}"


def _save(figure, output_path: Path) -> None:
    """Save one tight PNG and create only its immediate output directory."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
