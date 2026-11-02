"""Load and validate transformed behavior-classification datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd

DatasetSplit: TypeAlias = tuple[np.ndarray, np.ndarray, tuple[str, ...]]
BehaviorDatasets: TypeAlias = dict[str, DatasetSplit]


def build_feature_names() -> list[str]:
    """Build the ordered 141-feature contract produced by preprocessing.

    Returns:
        Feature names in exactly the same order as the columns in each NPZ
        feature matrix.

    Raises:
        AssertionError: If the generated contract is not exactly 141 unique
            feature names.
    """
    summary_features = [
        "torso_angle_mean",
        "torso_angle_std",
        "head_hip_compression_mean",
        "hip_ankle_vertical_diff_mean",
        "log_skeleton_spread_ratio_mean",
        "log_skeleton_spread_ratio_max",
        "sqrt_ground_speed_mean",
        "sqrt_ground_speed_max",
        "scale_speed_mean",
        "sqrt_combined_speed_mean",
        "sqrt_combined_speed_std",
        "body_speed_max",
        "sqrt_body_acceleration_max",
    ]
    step_metrics = [
        "torso_angle",
        "compression",
        "spread_ratio",
        "combined_speed",
        "hip_ankle",
        "ground_speed",
        "scale_speed",
        "body_speed",
    ]
    sequence_features = [
        f"step_{metric}_t{time_index}"
        for time_index in range(15)
        for metric in step_metrics
    ]
    final_window_features = [
        "final_lying_score",
        "final_1s_body_speed_mean",
        "final_1s_joint_motion_mean",
        "final_1s_joint_motion_std",
    ]
    quality_features = [
        "avg_keypoint_confidence",
        "missing_ankle_ratio",
        "missing_hip_ratio",
        "skeleton_jump_score",
    ]
    feature_names = (
        summary_features
        + sequence_features
        + final_window_features
        + quality_features
    )

    assert len(feature_names) == 141
    assert len(feature_names) == len(set(feature_names))
    return feature_names


def load_cleaned_split(path: Path) -> DatasetSplit:
    """Load one transformed dataset split without changing its values.

    Args:
        path: NPZ file containing feature array ``X`` and target array ``y``.

    Returns:
        Feature matrix, target vector, and the ordered NPZ key names.

    Raises:
        FileNotFoundError: If the NPZ file does not exist.
        KeyError: If either required array is absent.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing transformed dataset: {path}")

    with np.load(path, allow_pickle=False) as archive:
        keys = tuple(archive.files)
        missing_keys = {"X", "y"} - set(keys)
        if missing_keys:
            raise KeyError(
                f"{path.name} is missing required arrays: {sorted(missing_keys)}"
            )
        features = archive["X"]
        targets = archive["y"]
    return features, targets, keys


def load_behavior_datasets(data_dir: Path) -> BehaviorDatasets:
    """Load the fixed train, validation, and test NPZ artifacts.

    Args:
        data_dir: Directory containing the cleaned NPZ files.

    Returns:
        Mapping from split name to its feature matrix, target vector, and NPZ
        keys. Loading test here supports contract checks only; no prediction is
        performed by this function.
    """
    split_files = {
        "train": "train_cleaned.npz",
        "validation": "val_cleaned.npz",
        "test": "test_cleaned.npz",
    }
    return {
        split_name: load_cleaned_split(data_dir / filename)
        for split_name, filename in split_files.items()
    }


def validate_behavior_datasets(
    datasets: BehaviorDatasets,
    feature_names: list[str],
    class_ids: list[int],
) -> None:
    """Validate the modeling data contract without repairing any values.

    Args:
        datasets: Loaded train, validation, and test splits.
        feature_names: Ordered feature contract expected by every split.
        class_ids: Exact class identifiers expected in every target vector.

    Raises:
        ValueError: If a split violates its shape, value, key, feature, or class
            contract.
    """
    expected_splits = {"train", "validation", "test"}
    if set(datasets) != expected_splits:
        raise ValueError(f"Expected splits {expected_splits}, received {set(datasets)}")

    expected_classes = set(class_ids)
    for split_name, (features, targets, keys) in datasets.items():
        if keys != ("X", "y"):
            raise ValueError(f"Unexpected NPZ keys in {split_name}: {keys}")
        if features.ndim != 2 or targets.ndim != 1:
            raise ValueError(f"Invalid array dimensions in {split_name}")
        if features.shape[0] != targets.shape[0]:
            raise ValueError(f"Feature/target row mismatch in {split_name}")
        if features.shape[1] != len(feature_names):
            raise ValueError(
                f"{split_name} has {features.shape[1]} features; "
                f"expected {len(feature_names)}"
            )
        if set(np.unique(targets)) != expected_classes:
            raise ValueError(f"Unexpected class IDs in {split_name}")
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite feature values found in {split_name}")
        if pd.isna(targets).any():
            raise ValueError(f"Missing target values found in {split_name}")


def summarize_dataset_integrity(datasets: BehaviorDatasets) -> pd.DataFrame:
    """Create a compact integrity summary for notebook display.

    Args:
        datasets: Loaded behavior datasets.

    Returns:
        One row per split with sample count, feature count, and invalid-value
        counts.
    """
    rows = []
    for split_name, (features, _, _) in datasets.items():
        rows.append(
            {
                "split": split_name,
                "samples": features.shape[0],
                "features": features.shape[1],
                "missing_values": int(np.isnan(features).sum()),
                "infinite_values": int(np.isinf(features).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_class_distribution(
    datasets: BehaviorDatasets,
    class_mapping: dict[int, str],
) -> pd.DataFrame:
    """Summarize class counts and proportions for every dataset split.

    Args:
        datasets: Loaded behavior datasets.
        class_mapping: Mapping from numeric class ID to display name.

    Returns:
        Long-form table with split, class, count, and proportion columns.
    """
    rows = []
    for split_name, (_, targets, _) in datasets.items():
        class_ids, counts = np.unique(targets, return_counts=True)
        for class_id, count in zip(class_ids, counts, strict=True):
            rows.append(
                {
                    "split": split_name,
                    "class_name": class_mapping[int(class_id)],
                    "count": int(count),
                    "proportion": float(count / len(targets)),
                }
            )
    return pd.DataFrame(rows)


def _feature_row_hashes(features: np.ndarray) -> set[str]:
    """Hash exact feature rows without rounding or mutation."""
    return {
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in features
    }


def find_exact_row_overlap(datasets: BehaviorDatasets) -> pd.DataFrame:
    """Count exact transformed-row overlap between dataset splits.

    Args:
        datasets: Loaded behavior datasets.

    Returns:
        Symmetric matrix of overlap counts. The diagonal is zero because the
        diagnostic is intended only for cross-split comparisons.

    Notes:
        This is a limited duplicate-row diagnostic. It cannot prove that
        related windows from one source video are absent across splits.
    """
    split_names = list(datasets)
    hashes = {
        split_name: _feature_row_hashes(datasets[split_name][0])
        for split_name in split_names
    }
    overlap = pd.DataFrame(0, index=split_names, columns=split_names, dtype=int)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            overlap_count = len(hashes[left_name] & hashes[right_name])
            overlap.loc[left_name, right_name] = overlap_count
            overlap.loc[right_name, left_name] = overlap_count
    return overlap


def source_metadata_available(datasets: BehaviorDatasets) -> bool:
    """Return whether any cleaned split retains source/group metadata keys.

    Args:
        datasets: Loaded behavior datasets.

    Returns:
        ``True`` when at least one key other than ``X`` and ``y`` is present.
    """
    return any(
        key not in {"X", "y"}
        for _, _, keys in datasets.values()
        for key in keys
    )

