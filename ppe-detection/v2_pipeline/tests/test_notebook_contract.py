"""Structural tests for notebook ordering and test-data isolation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = V2_ROOT / "notebooks"


def notebook_code(name: str) -> str:
    """Return concatenated code-cell source from one notebook JSON file."""
    payload = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )


class NotebookContractTests(unittest.TestCase):
    """Prevent numbering drift, test leakage, and resolution fallback."""

    def test_expected_notebook_names_exist(self) -> None:
        """The core final and optional direct paths must follow calibration."""
        expected = {
            "06_small_object_resolution_calibration.ipynb",
            "07_final_training_and_test.ipynb",
            "08_direct_model_training_and_test.ipynb",
        }
        self.assertTrue(expected.issubset({path.name for path in NOTEBOOKS.glob("*.ipynb")}))
        self.assertFalse((NOTEBOOKS / "06_final_training_and_test.ipynb").exists())
        self.assertFalse((NOTEBOOKS / "07_direct_model_training_and_test.ipynb").exists())

    def test_resolution_calibration_has_no_test_access(self) -> None:
        """Notebook 06 must remain a train/validation-only selection stage."""
        source = notebook_code("06_small_object_resolution_calibration.ipynb")
        for forbidden in ('"test.txt"', 'split="test"', "test_manifest="):
            self.assertNotIn(forbidden, source)
        self.assertIn('split="val"', source)
        self.assertIn("resolution_selection_priority", source)

    def test_final_training_consumes_resolution_metadata(self) -> None:
        """Notebook 07 must apply resolution and the two-phase epoch policy."""
        source = notebook_code("07_final_training_and_test.ipynb")
        self.assertIn("SOURCE_RESOLUTION_RUN", source)
        self.assertIn('resolution_metadata.get("selected_imgsz")', source)
        self.assertIn('resolution_metadata.get("selected_batch")', source)
        self.assertIn("extract_best_epoch", source)
        self.assertIn("build_refit_training_manifest", source)
        self.assertIn('"val": False', source)
        self.assertIn('checkpoint_kind="last"', source)
        self.assertLess(source.index('split="test"'), len(source))
        self.assertGreater(source.index('split="test"'), source.index("refit_summary_df"))

    def test_direct_path_exposes_one_resolution_pair(self) -> None:
        """Notebook 08 must apply one explicit size/batch pair to both phases."""
        source = notebook_code("08_direct_model_training_and_test.ipynb")
        self.assertIn("TRAIN_IMGSZ", source)
        self.assertIn("TRAIN_BATCH", source)
        self.assertIn("DRY_RUN = True", source)
        self.assertIn("ALLOW_TEST_EVALUATION = False", source)


if __name__ == "__main__":
    unittest.main()
