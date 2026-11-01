"""Fixture tests for split manifests and final-test isolation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from src.config import get_experiment, load_pipeline_config
from src.dataset.manifests import (
    build_combined_training_manifest,
    build_development_training_manifest,
    build_original_training_manifest,
    build_refit_training_manifest,
    build_split_assignments,
    collect_yolo_samples,
    experiment_manifest,
    manifest_pair_content_sha256,
    read_manifest,
    validate_split_assignments,
    write_split_manifests,
    write_ultralytics_dataset_yaml,
)


class ManifestTests(unittest.TestCase):
    """Exercise the manifest workflow without loading the real dataset."""

    def setUp(self) -> None:
        """Create a tiny three-source YOLO fixture."""
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_dir.name)
        self.classes = {0: "person", 1: "helmet", 2: "vest", 3: "cleaning_coverall"}
        self.open_dir = self._make_source("open_source", [[0, 1], [0, 2]])
        self.factory_dir = self._make_source(
            "factory_source",
            [[0, 1], [0, 1], [0, 2], [0, 2], [0, 3], [0, 3]],
        )
        self.test_dir = self._make_source("test_source", [[0, 1], [0, 3]])

    def tearDown(self) -> None:
        """Release the fixture directory."""
        self.temporary_dir.cleanup()

    def test_split_is_deterministic_and_source_safe(self) -> None:
        """A fixed seed must produce stable, non-overlapping membership."""
        first = self._assign(seed=42)
        second = self._assign(seed=42)
        self.assertEqual(first["image_path"].tolist(), second["image_path"].tolist())
        self.assertTrue(
            first.loc[first["source"].eq("open_source"), "split"].eq("train").all()
        )
        self.assertTrue(
            first.loc[first["split"].eq("val"), "source"].eq("factory_source").all()
        )
        self.assertTrue(
            first.loc[first["split"].eq("test"), "source"].eq("test_source").all()
        )

    def test_candidate_yaml_omits_test_and_final_yaml_includes_it(self) -> None:
        """Only final evaluation may receive the untouched test manifest."""
        assignments = self._assign(seed=42)
        manifests = write_split_manifests(assignments, self.root / "manifests")
        candidate_yaml = write_ultralytics_dataset_yaml(
            self.root / "candidate.yaml",
            manifests["train_original"],
            manifests["val"],
            self.classes,
        )
        final_yaml = write_ultralytics_dataset_yaml(
            self.root / "final.yaml",
            manifests["train_original"],
            manifests["val"],
            self.classes,
            test_manifest=manifests["test"],
        )
        self.assertNotIn("test", yaml.safe_load(candidate_yaml.read_text(encoding="utf-8")))
        self.assertIn("test", yaml.safe_load(final_yaml.read_text(encoding="utf-8")))
        candidate_payload = yaml.safe_load(candidate_yaml.read_text(encoding="utf-8"))
        self.assertTrue(Path(candidate_payload["train"]).is_absolute())
        self.assertTrue(Path(candidate_payload["train"]).is_file())

    def test_pair_content_hash_covers_labels(self) -> None:
        """A label-only edit must change the frozen data fingerprint."""
        assignments = self._assign(seed=42)
        manifests = write_split_manifests(assignments, self.root / "manifests")
        original_digest = manifest_pair_content_sha256(manifests["test"])
        test_label = next((self.test_dir / "labels").glob("*.txt"))
        test_label.write_text("0 0.5 0.5 0.20 0.20\n", encoding="utf-8")
        self.assertNotEqual(
            original_digest,
            manifest_pair_content_sha256(manifests["test"]),
        )

    def test_two_manifests_cover_four_experiments(self) -> None:
        """A/B must share original data and C/D must share combined data."""
        assignments = self._assign(seed=42)
        manifests = write_split_manifests(assignments, self.root / "manifests")
        augmented_dir = self.root / "augmented" / "images"
        augmented_labels = self.root / "augmented" / "labels"
        augmented_dir.mkdir(parents=True)
        augmented_labels.mkdir(parents=True)
        image = augmented_dir / "ir__factory_source__000.png"
        Image.new("RGB", (16, 16), "gray").save(image)
        (augmented_labels / "ir__factory_source__000.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
        )
        combined = build_combined_training_manifest(
            manifests["train_original"],
            augmented_dir,
            self.root / "manifests" / "train_original_plus_offline.txt",
            {".png"},
        )
        config = load_pipeline_config(V2_ROOT)
        resolved = {
            name: experiment_manifest(
                get_experiment(config, name), manifests["train_original"], combined
            ).resolve()
            for name in config["experiments"]["experiments"]
        }
        self.assertEqual(resolved["exp_A_original_only"], resolved["exp_B_online_aug"])
        self.assertEqual(resolved["exp_C_offline_aug"], resolved["exp_D_full_pipeline"])
        self.assertGreater(
            len(read_manifest(combined)),
            len(read_manifest(manifests["train_original"])),
        )

    def test_development_refit_can_include_or_exclude_open_source(self) -> None:
        """Train-plus-val membership must honor the explicit source toggle."""
        assignments = self._assign(seed=42)
        manifests = write_split_manifests(assignments, self.root / "manifests")
        with_open = build_development_training_manifest(
            manifests["assignments"],
            self.root / "development_with_open.txt",
            include_open_source=True,
        )
        factory_only = build_development_training_manifest(
            manifests["assignments"],
            self.root / "development_factory_only.txt",
            include_open_source=False,
        )

        self.assertEqual(len(read_manifest(with_open)), 8)
        self.assertEqual(len(read_manifest(factory_only)), 6)
        self.assertTrue(
            all("factory_source" in str(path) for path in read_manifest(factory_only))
        )

    def test_calibration_manifest_keeps_validation_separate(self) -> None:
        """Calibration training must omit frozen factory validation rows."""
        assignments = self._assign(seed=42)
        manifests = write_split_manifests(assignments, self.root / "manifests")
        calibration = build_original_training_manifest(
            manifests["assignments"],
            self.root / "calibration_train.txt",
            include_open_source=False,
            include_validation=False,
        )
        expected = assignments.loc[
            assignments["split"].eq("train")
            & assignments["source"].eq("factory_source"),
            "image_path",
        ]
        self.assertEqual(
            {str(path) for path in read_manifest(calibration)},
            {str(Path(path).resolve()) for path in expected},
        )
        self.assertFalse(
            set(read_manifest(calibration)) & set(read_manifest(manifests["val"]))
        )

    def test_refit_yaml_requires_explicit_train_val_overlap(self) -> None:
        """Only the deliberate fixed-epoch refit may reuse one development list."""
        assignments = self._assign(seed=42)
        manifests = write_split_manifests(assignments, self.root / "manifests")
        development = build_development_training_manifest(
            manifests["assignments"],
            self.root / "development.txt",
            include_open_source=True,
        )
        with self.assertRaises(ValueError):
            write_ultralytics_dataset_yaml(
                self.root / "blocked.yaml",
                development,
                development,
                self.classes,
                test_manifest=manifests["test"],
            )

        refit_yaml = write_ultralytics_dataset_yaml(
            self.root / "refit.yaml",
            development,
            development,
            self.classes,
            test_manifest=manifests["test"],
            allow_train_val_overlap=True,
        )
        payload = yaml.safe_load(refit_yaml.read_text(encoding="utf-8"))
        self.assertEqual(payload["train"], payload["val"])
        self.assertEqual(payload["test"], str(manifests["test"].resolve()))

    def test_refit_manifest_preserves_selected_train_and_adds_validation(self) -> None:
        """Final refit must retain selected train membership and add clean val."""
        assignments = self._assign(seed=42)
        manifests = write_split_manifests(assignments, self.root / "manifests")
        selected_train = read_manifest(manifests["train_original"])
        validation = read_manifest(manifests["val"])

        refit = build_refit_training_manifest(
            manifests["train_original"],
            manifests["val"],
            self.root / "refit_train_plus_val.txt",
        )

        self.assertEqual(set(read_manifest(refit)), {*selected_train, *validation})
        self.assertFalse(set(read_manifest(refit)) & set(read_manifest(manifests["test"])))

    def test_overlap_guard_rejects_cross_split_path(self) -> None:
        """The same image path cannot appear in train and validation."""
        assignments = self._assign(seed=42)
        duplicated = assignments.iloc[[0]].copy()
        duplicated.loc[:, "split"] = "val"
        with self.assertRaises(ValueError):
            validate_split_assignments(
                pd.concat([assignments, duplicated], ignore_index=True)
            )

    def _assign(self, seed: int):
        """Collect fixture samples and return frozen assignments."""
        kwargs = {
            "class_names": self.classes,
            "image_extensions": {".png"},
            "allow_empty_labels": False,
        }
        open_samples = collect_yolo_samples(self.open_dir, "open_source", **kwargs)
        factory_samples = collect_yolo_samples(self.factory_dir, "factory_source", **kwargs)
        test_samples = collect_yolo_samples(self.test_dir, "test_source", **kwargs)
        assignments, _ = build_split_assignments(
            open_samples,
            factory_samples,
            test_samples,
            factory_train_ratio=0.8,
            factory_val_ratio=0.2,
            seed=seed,
        )
        return assignments

    def _make_source(self, name: str, class_rows: list[list[int]]) -> Path:
        """Create one tiny valid source lane for a unit test."""
        source = self.root / name
        images = source / "images"
        labels = source / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index, class_ids in enumerate(class_rows):
            stem = f"sample_{index:03d}"
            Image.new("RGB", (16, 16), "white").save(images / f"{stem}.png")
            label_text = "".join(
                f"{class_id} 0.5 0.5 0.25 0.25\n" for class_id in class_ids
            )
            (labels / f"{stem}.txt").write_text(label_text, encoding="utf-8")
        return source


if __name__ == "__main__":
    unittest.main()
