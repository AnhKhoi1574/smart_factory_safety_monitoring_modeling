"""Generate deterministic, non-geometric CCTV augmentations from a manifest."""

from __future__ import annotations

import io
import random
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.dataset.manifests import image_to_label_path, read_manifest


Transform = Callable[[Image.Image, random.Random], Image.Image]


def generate_offline_augmentations(
    train_manifest: Path,
    output_dir: Path,
    ratios: dict[str, float],
    *,
    seed: int,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Generate configured train-only augmentations and copy unchanged labels.

    Args:
        train_manifest: Frozen original training image paths.
        output_dir: Root containing generated `images/` and `labels/`.
        ratios: Ratios keyed by `ir`, `sunlight`, and `blur_compression`.
        seed: Deterministic selection and transform seed.
        overwrite: Whether this call may replace its exact generated targets.

    Returns:
        One concise report row per generated image-label pair.

    Raises:
        KeyError: If an unknown transform name is configured.
        FileExistsError: If a target exists and overwrite is disabled.
        FileNotFoundError: If a source label is missing.

    Side Effects:
        Writes only new augmented images and copied labels under `output_dir`.
        Original source files are never modified.
    """
    transforms: dict[str, Transform] = {
        "ir": _transform_ir,
        "sunlight": _transform_sunlight,
        "blur_compression": _transform_blur_compression,
    }
    unknown = set(ratios) - set(transforms)
    if unknown:
        raise KeyError(f"Unknown offline augmentation(s): {sorted(unknown)}")

    image_paths = read_manifest(train_manifest)
    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, Path, Path, Path, Path, Transform, random.Random]] = []
    for transform_index, transform_name in enumerate(sorted(ratios)):
        ratio = float(ratios[transform_name])
        if ratio <= 0:
            continue
        rng = random.Random(seed + (transform_index + 1) * 10_003)
        sample_count = min(len(image_paths), max(1, round(len(image_paths) * ratio)))
        selected = sorted(rng.sample(image_paths, sample_count), key=str)
        for image_path in selected:
            label_path = image_to_label_path(image_path)
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for augmentation source: {label_path}")
            source_name = image_path.parent.parent.name
            target_stem = f"{transform_name}__{source_name}__{image_path.stem}"
            target_image = images_dir / f"{target_stem}.jpg"
            target_label = labels_dir / f"{target_stem}.txt"
            plan.append(
                (
                    transform_name,
                    image_path,
                    label_path,
                    target_image,
                    target_label,
                    transforms[transform_name],
                    rng,
                )
            )

    occupied = [
        str(path)
        for row in plan
        for path in (row[3], row[4])
        if path.exists()
    ]
    if occupied and not overwrite:
        raise FileExistsError(
            "Offline augmentation targets already exist; enable overwrite only "
            f"for intentional regeneration: {occupied[:5]}"
        )

    rows = []
    for transform_name, image_path, label_path, target_image, target_label, transform, rng in plan:
        with Image.open(image_path) as image:
            augmented = transform(image.convert("RGB"), rng)
        augmented.save(target_image, format="JPEG", quality=92, optimize=True)
        shutil.copy2(label_path, target_label)
        rows.append(
            {
                "augmentation_type": transform_name,
                "status": "generated",
                "original_image_path": str(image_path),
                "original_label_path": str(label_path),
                "augmented_image_path": str(target_image.resolve()),
                "augmented_label_path": str(target_label.resolve()),
            }
        )
    return pd.DataFrame(rows)


def summarize_augmentations(report: pd.DataFrame) -> pd.DataFrame:
    """Return one generated-count row per transform."""
    if report.empty:
        return pd.DataFrame(columns=["augmentation_type", "generated_images"])
    return (
        report.groupby("augmentation_type", as_index=False)
        .agg(generated_images=("augmented_image_path", "count"))
        .sort_values("augmentation_type", kind="stable")
        .reset_index(drop=True)
    )


def _transform_ir(image: Image.Image, rng: random.Random) -> Image.Image:
    """Create a mild high-contrast monochrome CCTV/IR simulation."""
    grayscale = ImageOps.grayscale(image)
    contrast = ImageEnhance.Contrast(grayscale).enhance(rng.uniform(1.05, 1.25))
    return ImageOps.colorize(contrast, black=(4, 10, 16), white=(225, 238, 242))


def _transform_sunlight(image: Image.Image, rng: random.Random) -> Image.Image:
    """Add a bounded glare field without moving any object geometry."""
    array = np.asarray(image, dtype=np.float32)
    height, width = array.shape[:2]
    center_x = rng.uniform(0.15, 0.85) * width
    center_y = rng.uniform(0.05, 0.55) * height
    sigma = rng.uniform(0.20, 0.38) * max(width, height)
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    glare = np.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2 * sigma**2))
    strength = rng.uniform(35.0, 75.0)
    augmented = np.clip(array + glare[..., None] * strength, 0, 255).astype(np.uint8)
    return Image.fromarray(augmented, mode="RGB")


def _transform_blur_compression(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply mild optical blur followed by deterministic JPEG degradation."""
    blurred = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.4)))
    buffer = io.BytesIO()
    blurred.save(buffer, format="JPEG", quality=rng.randint(48, 72))
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB").copy()
