"""Tests for the small config and artifact helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from src.artifacts import create_run, read_run_metadata, runtime_metadata, write_run_metadata
from src.config import (
    _validate_training_config,
    get_experiment,
    load_pipeline_config,
    online_augmentation_args,
    resolve_final_experiment,
)


class ConfigAndArtifactTests(unittest.TestCase):
    """Verify fixed experiment semantics and immutable run directories."""

    def test_experiment_matrix_uses_two_dataset_variants(self) -> None:
        """A/B and C/D must share data while online augmentation differs."""
        config = load_pipeline_config(V2_ROOT)
        a = get_experiment(config, "exp_A_original_only")
        b = get_experiment(config, "exp_B_online_aug")
        c = get_experiment(config, "exp_C_offline_aug")
        d = get_experiment(config, "exp_D_full_pipeline")
        self.assertEqual(a["dataset_variant"], b["dataset_variant"])
        self.assertEqual(c["dataset_variant"], d["dataset_variant"])
        self.assertFalse(a["online_augmentation"])
        self.assertTrue(b["online_augmentation"])
        self.assertFalse(c["online_augmentation"])
        self.assertTrue(d["online_augmentation"])

    def test_disabled_online_augmentation_is_neutral(self) -> None:
        """Every configured online transform must receive a neutral value."""
        config = load_pipeline_config(V2_ROOT)
        disabled = online_augmentation_args(config["augmentation"], enabled=False)
        self.assertTrue(disabled)
        self.assertTrue(all(value == 0 for value in disabled.values()))

    def test_resolution_profiles_are_bounded_and_helmet_first(self) -> None:
        """Calibration should expose only the three reviewed size/batch pairs."""
        training = load_pipeline_config(V2_ROOT)["training"]
        self.assertEqual(
            training["resolution_profiles"],
            [
                {"name": "img640", "imgsz": 640, "batch": 32},
                {"name": "img960", "imgsz": 960, "batch": 16},
                {"name": "img1280", "imgsz": 1280, "batch": 8},
            ],
        )
        self.assertEqual(training["resolution_selection_priority"][0], "helmet_recall")

    def test_resolution_profiles_reject_duplicate_names(self) -> None:
        """Duplicate profile names must fail before any training starts."""
        invalid = {
            "resolution_epochs": 10,
            "resolution_profiles": [
                {"name": "same", "imgsz": 640, "batch": 16},
                {"name": "same", "imgsz": 960, "batch": 8},
            ],
            "resolution_selection_priority": [
                "helmet_recall",
                "helmet_map50_95",
                "overall_map50_95",
                "fps",
            ],
        }
        with self.assertRaisesRegex(ValueError, "unique"):
            _validate_training_config(invalid)

    def test_final_experiment_defaults_to_ablation_selection(self) -> None:
        """Final training should follow Notebook 05 when no override is set."""
        config = load_pipeline_config(V2_ROOT)
        experiment, source = resolve_final_experiment(
            config,
            "exp_B_online_aug",
        )
        self.assertEqual(experiment, "exp_B_online_aug")
        self.assertEqual(source, "ablation")

    def test_final_experiment_can_explicitly_override_with_full_pipeline(self) -> None:
        """An explicit D override should be validated and auditable."""
        config = load_pipeline_config(V2_ROOT)
        experiment, source = resolve_final_experiment(
            config,
            "exp_A_original_only",
            override="exp_D_full_pipeline",
        )
        self.assertEqual(experiment, "exp_D_full_pipeline")
        self.assertEqual(source, "manual_override")

    def test_final_experiment_rejects_unknown_override(self) -> None:
        """A typo in the manual override must fail before test evaluation."""
        config = load_pipeline_config(V2_ROOT)
        with self.assertRaises(KeyError):
            resolve_final_experiment(
                config,
                "exp_A_original_only",
                override="exp_unknown",
            )

    def test_run_creation_never_overwrites(self) -> None:
        """An identical run ID must fail instead of overwriting artifacts."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            moment = datetime(2026, 8, 25, 7, 30, tzinfo=timezone.utc)
            run = create_run(
                root,
                "candidate-models",
                "YOLO sweep: Windows safe",
                42,
                timestamp=moment,
                git_sha="abc1234",
            )
            write_run_metadata(run, {"status": "created"})
            self.assertEqual(read_run_metadata(run)["status"], "created")
            self.assertNotIn(":", run.run_id)
            self.assertNotIn(" ", run.run_id)
            with self.assertRaises(FileExistsError):
                create_run(
                    root,
                    "candidate-models",
                    "YOLO sweep: Windows safe",
                    42,
                    timestamp=moment,
                    git_sha="abc1234",
                )

    def test_runtime_metadata_records_accelerator_contract(self) -> None:
        """Run metadata must expose package and accelerator reproducibility fields."""
        metadata = runtime_metadata(V2_ROOT)
        self.assertIn("torch", metadata["packages"])
        self.assertIn("accelerator", metadata)
        self.assertIn("cuda_available", metadata["accelerator"])
        self.assertIn("gpu_devices", metadata["accelerator"])


if __name__ == "__main__":
    unittest.main()
