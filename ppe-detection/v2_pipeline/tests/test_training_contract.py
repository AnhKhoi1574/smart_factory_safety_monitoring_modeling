"""Tests for training dry-run and validation-only ranking behavior."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from unittest.mock import patch

from src.training.evaluate_yolo import extract_best_epoch
from src.training.train_yolo import (
    _release_training_resources,
    build_train_args,
    format_progress_event,
    rank_results,
    retain_training_checkpoint,
    run_training_specs,
)


class TrainingContractTests(unittest.TestCase):
    """Verify structural training behavior without importing Ultralytics."""

    def test_runtime_loader_settings_are_forwarded(self) -> None:
        """Worker and cache policy should reach Ultralytics unchanged."""
        config = {
            "candidate_epochs": 50,
            "imgsz": 640,
            "batch": 32,
            "device": 0,
            "workers": 6,
            "cache": False,
            "patience": 20,
            "seed": 42,
            "amp": True,
        }
        args = build_train_args(config, "candidate_epochs")
        self.assertEqual(args["batch"], 32)
        self.assertEqual(args["workers"], 6)
        self.assertFalse(args["cache"])

    def test_training_resources_are_released_between_models(self) -> None:
        """Both persistent loaders and trainer references should be released."""

        class FakeIterator:
            """Record whether the private PyTorch worker shutdown hook ran."""

            def __init__(self) -> None:
                self.shutdown_called = False

            def _shutdown_workers(self) -> None:
                self.shutdown_called = True

        class FakeLoader:
            """Expose the iterator shape used by Ultralytics InfiniteDataLoader."""

            def __init__(self) -> None:
                self.iterator = FakeIterator()

        class Holder:
            """Provide mutable attributes matching the small trainer contract."""

            pass

        train_loader = FakeLoader()
        test_loader = FakeLoader()
        validator = Holder()
        validator.dataloader = test_loader
        trainer = Holder()
        trainer.train_loader = train_loader
        trainer.test_loader = test_loader
        trainer.validator = validator
        trainer.optimizer = object()
        trainer.model = object()
        model = Holder()
        model.trainer = trainer
        model.model = object()
        model.ckpt = object()
        model.metrics = object()

        _release_training_resources(model)

        self.assertTrue(train_loader.iterator.shutdown_called)
        self.assertTrue(test_loader.iterator.shutdown_called)
        self.assertIsNone(validator.dataloader)
        self.assertIsNone(trainer.train_loader)
        self.assertIsNone(trainer.test_loader)
        self.assertIsNone(trainer.optimizer)
        self.assertIsNone(model.trainer)
        self.assertIsNone(model.model)

    def test_dry_run_does_not_require_ultralytics(self) -> None:
        """A valid runtime YAML should produce one dry-run result row."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset_yaml = root / "dataset.yaml"
            dataset_yaml.write_text("train: train.txt\nval: val.txt\n", encoding="utf-8")
            events = []
            results = run_training_specs(
                [
                    {
                        "run_name": "candidate-yolo11n",
                        "model_name": "yolo11n.pt",
                        "data_yaml": dataset_yaml,
                        "online_augmentation": True,
                    }
                ],
                root / "models",
                {"epochs": 1},
                dry_run=True,
                progress_callback=events.append,
            )
            self.assertEqual(results.loc[0, "status"], "dry_run")
            self.assertFalse((root / "models" / "_work").exists())
            self.assertEqual([event["event"] for event in events], ["model_start", "model_end"])
            self.assertEqual(events[-1]["status"], "dry_run")

    def test_progress_markdown_contains_epoch_and_metrics(self) -> None:
        """Epoch progress should remain compact and readable in one display."""
        rendered = format_progress_event(
            {
                "event": "epoch_end",
                "model_index": 1,
                "model_total": 6,
                "model_name": "yolov8n.pt",
                "epoch": 5,
                "epochs": 50,
                "box_loss": 1.2,
                "recall": 0.79,
                "map50": 0.82,
                "elapsed_seconds": 125,
            }
        )
        self.assertIn("[1/6]", rendered)
        self.assertIn("05/50", rendered)
        self.assertIn("box=1.2000", rendered)
        self.assertIn("R=0.7900", rendered)
        self.assertIn("elapsed=02:05", rendered)

    def test_refit_progress_hides_unavailable_validation_metrics(self) -> None:
        """A val-disabled refit should not render placeholder metric zeroes."""
        rendered = format_progress_event(
            {
                "event": "epoch_end",
                "model_index": 1,
                "model_total": 1,
                "model_name": "yolo12n.pt",
                "epoch": 20,
                "epochs": 40,
                "validation_enabled": False,
                "box_loss": 0.9,
                "recall": 0.0,
                "map50": 0.0,
            }
        )
        self.assertIn("box=0.9000", rendered)
        self.assertNotIn("R=", rendered)
        self.assertNotIn("mAP50", rendered)

    def test_best_epoch_uses_first_maximum_map50_95(self) -> None:
        """Calibration should lock the first epoch with maximum mAP50-95."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            pd.DataFrame(
                [
                    {"epoch": 1, "metrics/recall(B)": 0.5, "metrics/mAP50(B)": 0.6, "metrics/mAP50-95(B)": 0.30},
                    {"epoch": 2, "metrics/recall(B)": 0.7, "metrics/mAP50(B)": 0.8, "metrics/mAP50-95(B)": 0.55},
                    {"epoch": 3, "metrics/recall(B)": 0.8, "metrics/mAP50(B)": 0.9, "metrics/mAP50-95(B)": 0.55},
                ]
            ).to_csv(run_dir / "results.csv", index=False)

            selected = extract_best_epoch(run_dir)

            self.assertEqual(selected["best_epoch"], 2)
            self.assertAlmostEqual(selected["map50_95"], 0.55)
            self.assertAlmostEqual(selected["recall"], 0.7)

    def test_failed_training_retains_only_log_tail(self) -> None:
        """Only a bounded raw-log excerpt should survive a failed model run."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset_yaml = root / "dataset.yaml"
            dataset_yaml.write_text("train: train.txt\nval: val.txt\n", encoding="utf-8")

            def fail_training(*args, captured_log, **kwargs):
                """Populate a synthetic long log before raising an expected error."""
                captured_log.write("\n".join(f"line-{index}" for index in range(10)))
                raise RuntimeError("synthetic failure")

            with patch("src.training.train_yolo._train_one_spec", side_effect=fail_training):
                results = run_training_specs(
                    [{"run_name": "broken", "model_name": "broken.pt", "data_yaml": dataset_yaml}],
                    root / "models",
                    {"epochs": 1},
                    failure_log_lines=3,
                )
            self.assertEqual(results.loc[0, "status"], "failed")
            self.assertEqual(
                results.loc[0, "failure_log_excerpt"],
                "line-7\nline-8\nline-9",
            )

    def test_ranking_is_lexicographic_and_failure_last(self) -> None:
        """Recall takes priority over mAP and failed rows remain last."""
        results = pd.DataFrame(
            [
                {"model_name": "high-map", "status": "trained", "recall": 0.80, "map50": 0.95, "map50_95": 0.7, "fps": 30, "model_size_mb": 5},
                {"model_name": "high-recall", "status": "trained", "recall": 0.85, "map50": 0.80, "map50_95": 0.6, "fps": 20, "model_size_mb": 6},
                {"model_name": "failed", "status": "failed", "recall": 0.99, "map50": 0.99, "map50_95": 0.9, "fps": 50, "model_size_mb": 4},
            ]
        )
        ranked = rank_results(
            results,
            ["recall", "map50", "map50_95", "fps", "model_size_mb"],
        )
        self.assertEqual(ranked.iloc[0]["model_name"], "high-recall")
        self.assertEqual(ranked.iloc[-1]["model_name"], "failed")

    def test_resolution_ranking_prioritizes_helmet_recall(self) -> None:
        """A larger overall mAP must not hide lower helmet recall."""
        results = pd.DataFrame(
            [
                {
                    "profile": "img640",
                    "status": "trained",
                    "helmet_recall": 0.70,
                    "helmet_map50_95": 0.50,
                    "overall_map50_95": 0.70,
                    "fps": 80,
                },
                {
                    "profile": "img960",
                    "status": "trained",
                    "helmet_recall": 0.76,
                    "helmet_map50_95": 0.48,
                    "overall_map50_95": 0.66,
                    "fps": 50,
                },
            ]
        )
        ranked = rank_results(
            results,
            ["helmet_recall", "helmet_map50_95", "overall_map50_95", "fps"],
        )
        self.assertEqual(ranked.iloc[0]["profile"], "img960")

    def test_fixed_epoch_refit_can_retain_last_checkpoint(self) -> None:
        """A train-plus-val refit should retain last.pt, not select best.pt."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            last_checkpoint = root / "work" / "weights" / "last.pt"
            last_checkpoint.parent.mkdir(parents=True)
            last_checkpoint.write_bytes(b"last-checkpoint")
            results = pd.DataFrame(
                [
                    {
                        "status": "trained",
                        "best_weights_path": "",
                        "last_weights_path": str(last_checkpoint),
                    }
                ]
            )

            retained = retain_training_checkpoint(
                results,
                root / "models" / "direct_final_last.pt",
                checkpoint_kind="last",
            )

            self.assertEqual(retained.read_bytes(), b"last-checkpoint")


if __name__ == "__main__":
    unittest.main()
