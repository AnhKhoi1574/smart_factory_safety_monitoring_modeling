"""Tests for class-ID and empty-label validation policy."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from src.dataset.validate_yolo_dataset import validate_source_directory


class ValidationTests(unittest.TestCase):
    """Verify the fixed four-class and empty-label source contract."""

    def test_valid_four_class_rows_are_accepted(self) -> None:
        """Class IDs 0-3 with valid boxes must pass validation."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = self._source(Path(temporary_dir), "\n".join(
                f"{class_id} 0.5 0.5 0.2 0.2" for class_id in range(4)
            ))
            report = validate_source_directory(source, "fixture")
            self.assertEqual(report.loc[0, "status"], "valid")
            self.assertEqual(report.loc[0, "num_objects"], 4)

    def test_unknown_class_is_rejected(self) -> None:
        """Class IDs outside 0-3 must produce an invalid audit row."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = self._source(Path(temporary_dir), "4 0.5 0.5 0.2 0.2")
            report = validate_source_directory(source, "fixture")
            self.assertEqual(report.loc[0, "status"], "invalid")
        self.assertIn("class_id 4 is not allowed", report.loc[0, "errors"])

    def test_empty_label_follows_explicit_policy(self) -> None:
        """An empty label is invalid by default and valid only when enabled."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = self._source(Path(temporary_dir), "")
            strict = validate_source_directory(source, "fixture", allow_empty_labels=False)
            allowed = validate_source_directory(source, "fixture", allow_empty_labels=True)
            self.assertEqual(strict.loc[0, "status"], "invalid")
            self.assertEqual(allowed.loc[0, "status"], "valid")

    def test_duplicate_annotation_row_is_rejected(self) -> None:
        """Equivalent YOLO rows must be reported instead of counted twice."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = self._source(
                Path(temporary_dir),
                "0 0.5 0.5 0.2 0.2\n0 .5 .5 .2 .2\n",
            )
            report = validate_source_directory(source, "fixture")
            self.assertEqual(report.loc[0, "status"], "invalid")
            self.assertEqual(report.loc[0, "num_objects"], 1)
            self.assertIn(
                "Line 2: duplicate annotation row; matches line 1",
                report.loc[0, "errors"],
            )

    def _source(self, root: Path, label_text: str) -> Path:
        """Create one image-label source lane for a validation test."""
        images = root / "images"
        labels = root / "labels"
        images.mkdir()
        labels.mkdir()
        Image.new("RGB", (32, 32), "white").save(images / "sample.png")
        (labels / "sample.txt").write_text(label_text, encoding="utf-8")
        return root


if __name__ == "__main__":
    unittest.main()
