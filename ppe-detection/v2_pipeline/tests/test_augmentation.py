"""Fixture tests for deterministic non-geometric offline augmentation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from src.augmentation.offline import generate_offline_augmentations
from src.dataset.manifests import write_manifest


class OfflineAugmentationTests(unittest.TestCase):
    """Verify generated samples are deterministic and preserve label bytes."""

    def test_generation_writes_only_new_images_and_copied_labels(self) -> None:
        """Three transforms should create three paired fixture outputs."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            images = root / "factory_source" / "images"
            labels = root / "factory_source" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            source_paths = []
            for index in range(2):
                image_path = images / f"sample_{index}.png"
                Image.new("RGB", (24, 24), (100 + index, 120, 140)).save(image_path)
                (labels / f"sample_{index}.txt").write_text(
                    "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
                )
                source_paths.append(image_path)
            manifest = write_manifest(root / "train.txt", source_paths)

            report = generate_offline_augmentations(
                manifest,
                root / "augmented",
                {"ir": 0.5, "sunlight": 0.5, "blur_compression": 0.5},
                seed=42,
            )

            self.assertEqual(len(report), 3)
            self.assertEqual(set(report["augmentation_type"]), {"ir", "sunlight", "blur_compression"})
            for row in report.itertuples(index=False):
                original_label = Path(row.original_label_path).read_bytes()
                augmented_label = Path(row.augmented_label_path).read_bytes()
                self.assertEqual(original_label, augmented_label)
                self.assertTrue(Path(row.augmented_image_path).is_file())


if __name__ == "__main__":
    unittest.main()
