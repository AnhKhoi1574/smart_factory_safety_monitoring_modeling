"""Lightweight callback integration test for the pinned Ultralytics runtime."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from src.training.train_yolo import run_training_specs


ULTRALYTICS_AVAILABLE = importlib.util.find_spec("ultralytics") is not None


@unittest.skipUnless(ULTRALYTICS_AVAILABLE, "Ultralytics is not installed")
class UltralyticsIntegrationTests(unittest.TestCase):
    """Verify callback and logger behavior without training a real model."""

    def test_callback_events_and_logger_restoration(self) -> None:
        """Raw output stays captured while epoch events reach the UI callback."""
        from ultralytics.utils import LOGGER

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset_yaml = root / "dataset.yaml"
            dataset_yaml.write_text(
                "train: train.txt\nval: val.txt\n",
                encoding="utf-8",
            )
            native_run = root / "models" / "_work" / "fake"
            events: list[dict] = []
            outer_stdout = io.StringIO()
            logger_was_disabled = LOGGER.disabled

            with (
                patch(
                    "src.training.train_yolo._load_model",
                    return_value=_FakeModel(native_run),
                ),
                contextlib.redirect_stdout(outer_stdout),
            ):
                results = run_training_specs(
                    [
                        {
                            "run_name": "fake",
                            "model_name": "fake.pt",
                            "data_yaml": dataset_yaml,
                        }
                    ],
                    root / "models",
                    {"epochs": 50},
                    progress_callback=events.append,
                )

            self.assertEqual(
                results.loc[0, "status"],
                "trained",
                msg=results.loc[0].to_dict(),
            )
            self.assertEqual(
                [event["event"] for event in events],
                ["model_start", "epoch_end", "model_end"],
            )
            self.assertEqual(events[1]["epoch"], 5)
            self.assertEqual(events[1]["recall"], 0.79)
            self.assertNotIn("RAW_ULTRALYTICS_OUTPUT", outer_stdout.getvalue())
            self.assertEqual(LOGGER.disabled, logger_was_disabled)


class _FakeTrainer:
    """Expose the trainer fields consumed by the epoch callback."""

    epoch = 4
    epochs = 50
    tloss = [1.2, 0.5, 0.9]
    metrics = {
        "metrics/precision(B)": 0.77,
        "metrics/recall(B)": 0.79,
        "metrics/mAP50(B)": 0.82,
        "metrics/mAP50-95(B)": 0.51,
    }
    train_time_start = time.time()

    @staticmethod
    def label_loss_items(values: list[float]) -> dict[str, float]:
        """Map synthetic losses to Ultralytics detection metric names."""
        return dict(
            zip(
                ("train/box_loss", "train/cls_loss", "train/dfl_loss"),
                values,
            )
        )


class _FakeResult:
    """Provide the save directory expected by the training wrapper."""

    def __init__(self, save_dir: Path) -> None:
        """Store the fake native run directory."""
        self.save_dir = save_dir


class _FakeModel:
    """Invoke a registered callback instead of training a real model."""

    def __init__(self, save_dir: Path) -> None:
        """Initialize fake model state."""
        self.save_dir = save_dir
        self.callback = None

    def add_callback(self, event: str, callback) -> None:
        """Capture the single epoch callback registered by the wrapper."""
        if event != "on_fit_epoch_end":
            raise AssertionError(f"Unexpected callback event: {event}")
        self.callback = callback

    def train(self, **kwargs) -> _FakeResult:
        """Emit raw text and one synthetic completed-epoch event."""
        print("RAW_ULTRALYTICS_OUTPUT")
        if self.callback is None:
            raise AssertionError("Epoch callback was not registered")
        self.callback(_FakeTrainer())
        return _FakeResult(self.save_dir)


if __name__ == "__main__":
    unittest.main()
