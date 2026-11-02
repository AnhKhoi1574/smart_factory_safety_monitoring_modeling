"""Standalone script to extract the 141 pre-engineered posture and locomotion features
(excluding track_gap_count and valid_frame_ratio) from raw exported keypoint windows
across multiple sub-datasets, merge them, split them using a database-independent
overlap track clustering method, and save them into 3 distinct split .npz files.

Usage:
    python extract_engineered_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

import math
from statistics import fmean, pstdev
from typing import Any, Iterable

behaviors_datasets_dir = Path(__file__).resolve().parent

CANONICAL_FPS = 24
FEATURE_SCHEMA_VERSION = "v1"

COCO = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_ankle": 15,
    "right_ankle": 16
}

STEP_METRICS = (
    "torso_angle",
    "compression",
    "spread_ratio",
    "combined_speed",
    "hip_ankle",
    "ground_speed",
    "scale_speed",
    "body_speed",
)


def ramp(value: float, low: float, high: float) -> float:
    """Linearly map a raw value into the closed interval zero to one."""
    if high <= low:
        raise ValueError("Transform high bound must be greater than low bound")
    return min(1.0, max(0.0, (value - low) / (high - low)))


def _mean(values: Iterable[float | None], default: float = 0.0) -> float:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return fmean(clean) if clean else default


def _std(values: Iterable[float | None]) -> float:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return pstdev(clean) if len(clean) > 1 else 0.0


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _point(frame: dict[str, Any], names: tuple[str, ...], threshold: float) -> tuple[float, float] | None:
    points: list[tuple[float, float]] = []
    for name in names:
        index = COCO[name]
        scores = frame.get("keypoint_scores", [])
        keypoints = frame.get("keypoints", [])
        if index >= len(scores) or index >= len(keypoints) or float(scores[index]) < threshold:
            continue
        point = keypoints[index]
        points.append((float(point[0]), float(point[1])))
    if len(points) != len(names):
        return None
    return (_mean(point[0] for point in points), _mean(point[1] for point in points))


def _frame_geometry(frame: dict[str, Any] | None, keypoint_threshold: float) -> dict[str, Any]:
    if frame is None:
        return {name: None for name in (*STEP_METRICS, "body_center", "ground_point", "bbox_height", "body_size", "joint_positions")}
    bbox = [float(value) for value in frame["bbox"]]
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    diagonal = math.hypot(width, height)
    shoulder = _point(frame, ("left_shoulder", "right_shoulder"), keypoint_threshold)
    hip = _point(frame, ("left_hip", "right_hip"), keypoint_threshold)
    ankles = _point(frame, ("left_ankle", "right_ankle"), keypoint_threshold)
    nose = _point(frame, ("nose",), keypoint_threshold)
    torso_length = _distance(shoulder, hip) if shoulder and hip else 0.0
    body_size = diagonal or height or torso_length or 1.0
    torso = None
    if shoulder and hip:
        torso = math.degrees(math.atan2(abs(shoulder[1] - hip[1]), abs(shoulder[0] - hip[0])))
    compression = abs(nose[1] - hip[1]) / height if nose and hip and height else None
    hip_ankle = abs(hip[1] - ankles[1]) / body_size if hip and ankles else None
    valid_points = [
        (float(point[0]), float(point[1]))
        for point, score in zip(frame.get("keypoints", []), frame.get("keypoint_scores", []), strict=False)
        if float(score) >= keypoint_threshold
    ]
    if len(valid_points) >= 2:
        spread_height = max(point[1] for point in valid_points) - min(point[1] for point in valid_points)
        spread = (max(point[0] for point in valid_points) - min(point[0] for point in valid_points)) / spread_height if spread_height else width / height if height else 0.0
    else:
        spread = width / height if height else 0.0
    body_center = ((_mean((shoulder[0], hip[0])), _mean((shoulder[1], hip[1])))) if shoulder and hip else None
    ground = ankles or ((bbox[0] + bbox[2]) / 2.0, bbox[3])
    return {
        "torso_angle": torso,
        "compression": compression,
        "spread_ratio": spread,
        "hip_ankle": hip_ankle,
        "body_center": body_center,
        "ground_point": ground,
        "bbox_height": height,
        "body_size": body_size,
        "joint_positions": valid_points,
        "ground_speed": None,
        "scale_speed": None,
        "combined_speed": None,
        "body_speed": None,
    }


def _add_motion(metrics: list[dict[str, Any]], fps: int) -> None:
    delta_time = 1.0 / fps
    for index in range(1, len(metrics)):
        current = metrics[index]
        previous = metrics[index - 1]
        if current["ground_point"] and previous["ground_point"] and current["bbox_height"]:
            current["ground_speed"] = _distance(current["ground_point"], previous["ground_point"]) / (current["bbox_height"] * delta_time)
        if current["bbox_height"] and previous["bbox_height"]:
            current["scale_speed"] = abs(current["bbox_height"] - previous["bbox_height"]) / (current["bbox_height"] * delta_time)
        if current["ground_speed"] is not None or current["scale_speed"] is not None:
            current["combined_speed"] = (current["ground_speed"] or 0.0) + 0.45 * (current["scale_speed"] or 0.0)
        if current["body_center"] and previous["body_center"] and current["body_size"]:
            current["body_speed"] = _distance(current["body_center"], previous["body_center"]) / (current["body_size"] * delta_time)


def _interpolate(values: list[float | None], max_gap: int) -> tuple[list[float | None], list[str]]:
    result = list(values)
    provenance = ["observed" if value is not None else "missing" for value in values]
    index = 0
    while index < len(result):
        if result[index] is not None:
            index += 1
            continue
        start = index
        while index < len(result) and result[index] is None:
            index += 1
        end = index - 1
        gap = end - start + 1
        left = start - 1
        right = index
        if gap <= max_gap and left >= 0 and right < len(result) and result[left] is not None and result[right] is not None:
            for offset, target in enumerate(range(start, right), start=1):
                fraction = offset / (gap + 1)
                result[target] = float(result[left]) + (float(result[right]) - float(result[left])) * fraction
                provenance[target] = "interpolated"
    return result, provenance


def feature_columns() -> list[str]:
    """Return the exact stable external-model feature order for schema v1."""
    summary = [
        "torso_angle_mean", "torso_angle_std", "head_hip_compression_mean",
        "hip_ankle_vertical_diff_mean", "skeleton_spread_ratio_mean", "skeleton_spread_ratio_max",
        "ground_speed_mean", "ground_speed_max", "scale_speed_mean", "combined_speed_mean",
        "combined_speed_std", "body_speed_max", "body_acceleration_max",
    ]
    steps = [f"step_{name}_t{index}" for index in range(15) for name in STEP_METRICS]
    final = ["final_lying_score", "final_1s_body_speed_mean", "final_1s_joint_motion_mean", "final_1s_joint_motion_std"]
    quality = ["avg_keypoint_confidence", "missing_ankle_ratio", "valid_frame_ratio", "missing_hip_ratio", "track_gap_count", "skeleton_jump_score"]
    return [*summary, *steps, *final, *quality]


def extract_window_features(
    frames: list[dict[str, Any] | None],
    *,
    keypoint_threshold: float = 0.10,
    max_interpolation_gap: int = 6,
    min_valid_frame_ratio: float = 0.70,
) -> dict[str, Any]:
    """Extract all Group A-E features from one 60-frame track window."""
    if len(frames) != 60:
        raise ValueError("Feature windows must contain exactly 60 canonical frames")
    metrics = [_frame_geometry(frame, keypoint_threshold) for frame in frames]
    _add_motion(metrics, CANONICAL_FPS)
    valid_frame_ratio = sum(frame is not None for frame in frames) / len(frames)
    arrays: dict[str, list[float | None]] = {}
    provenance: dict[str, list[str]] = {}
    for name in STEP_METRICS:
        arrays[name], provenance[name] = _interpolate([item[name] for item in metrics], max_interpolation_gap)
    body_speeds = arrays["body_speed"]
    accelerations = [
        abs(float(current) - float(previous)) * CANONICAL_FPS
        if current is not None and previous is not None else None
        for previous, current in zip(body_speeds, body_speeds[1:], strict=False)
    ]
    raw: dict[str, float] = {
        "torso_angle_mean": _mean(arrays["torso_angle"]),
        "torso_angle_std": _std(arrays["torso_angle"]),
        "head_hip_compression_mean": _mean(arrays["compression"]),
        "hip_ankle_vertical_diff_mean": _mean(arrays["hip_ankle"]),
        "skeleton_spread_ratio_mean": _mean(arrays["spread_ratio"]),
        "skeleton_spread_ratio_max": max((value for value in arrays["spread_ratio"] if value is not None), default=0.0),
        "ground_speed_mean": _mean(arrays["ground_speed"]),
        "ground_speed_max": max((value for value in arrays["ground_speed"] if value is not None), default=0.0),
        "scale_speed_mean": _mean(arrays["scale_speed"]),
        "combined_speed_mean": _mean(arrays["combined_speed"]),
        "combined_speed_std": _std(arrays["combined_speed"]),
        "body_speed_max": max((value for value in body_speeds if value is not None), default=0.0),
        "body_acceleration_max": max((value for value in accelerations if value is not None), default=0.0),
    }
    for step, frame_index in enumerate(range(0, 60, 4)):
        for name in STEP_METRICS:
            raw[f"step_{name}_t{step}"] = float(arrays[name][frame_index] or 0.0)
    final_metrics = metrics[-24:]
    final_body = arrays["body_speed"][-24:]
    joint_motion: list[float] = []
    for previous, current in zip(final_metrics, final_metrics[1:], strict=False):
        previous_points = previous["joint_positions"]
        current_points = current["joint_positions"]
        if previous_points and current_points and len(previous_points) == len(current_points):
            joint_motion.append(_mean(_distance(first, second) for first, second in zip(previous_points, current_points, strict=True)) / max(current["body_size"], 1.0))
    final_torso = _mean(arrays["torso_angle"][-24:])
    final_spread = _mean(arrays["spread_ratio"][-24:])
    final_still = _mean(joint_motion)
    raw.update({
        "final_lying_score": _mean((1.0 - ramp(final_torso, 25.0, 75.0), ramp(final_spread, 0.8, 1.6), 1.0 - ramp(final_still, 0.01, 0.20))),
        "final_1s_body_speed_mean": _mean(final_body),
        "final_1s_joint_motion_mean": final_still,
        "final_1s_joint_motion_std": _std(joint_motion),
    })
    confidences = [float(score) for frame in frames if frame for score in frame.get("keypoint_scores", [])]
    missing_ankles = sum(
        frame is None or any(float(frame.get("keypoint_scores", [0.0] * 17)[index]) < keypoint_threshold for index in (15, 16))
        for frame in frames
    ) / len(frames)
    missing_hips = sum(
        frame is None or any(float(frame.get("keypoint_scores", [0.0] * 17)[index]) < keypoint_threshold for index in (11, 12))
        for frame in frames
    ) / len(frames)
    observed_indexes = [index for index, frame in enumerate(frames) if frame]
    gaps = sum((right - left) > 1 for left, right in zip(observed_indexes, observed_indexes[1:], strict=False))
    centers = [metric["body_center"] for metric in metrics]
    jumps = [
        _distance(current, previous) / max(metrics[index]["body_size"], 1.0)
        for index, (previous, current) in enumerate(zip(centers, centers[1:], strict=False), start=1)
        if previous and current
    ]
    raw.update({
        "avg_keypoint_confidence": _mean(confidences),
        "missing_ankle_ratio": missing_ankles,
        "valid_frame_ratio": valid_frame_ratio,
        "missing_hip_ratio": missing_hips,
        "track_gap_count": float(gaps),
        "skeleton_jump_score": max(jumps, default=0.0),
    })
    step_provenance = {
        f"step_{name}_t{step}": provenance[name][frame_index]
        for step, frame_index in enumerate(range(0, 60, 4))
        for name in STEP_METRICS
    }
    quality_status = "good" if valid_frame_ratio >= min_valid_frame_ratio else "low_quality"
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "raw": raw,
        "provenance": step_provenance,
        "quality": {
            "status": quality_status,
            "valid_frame_ratio": valid_frame_ratio,
            "observed_steps": sum(value == "observed" for value in step_provenance.values()),
            "interpolated_steps": sum(value == "interpolated" for value in step_provenance.values()),
            "missing_steps": sum(value == "missing" for value in step_provenance.values()),
        },
    }



def main():
    export_dir = behaviors_datasets_dir / "raw_data"
    
    # 3 distinct output files saved to features_extracted_data directory
    train_file = behaviors_datasets_dir / "features_extracted_data" / "train_split.npz"
    val_file = behaviors_datasets_dir / "features_extracted_data" / "val_split.npz"
    test_file = behaviors_datasets_dir / "features_extracted_data" / "test_split.npz"

    subdirs = ["fall", "no-fall", "run"]
    
    keypoints_list = []
    bboxes_list = []
    scores_list = []
    labels_list = []
    
    groups = []
    global_window_offset = 0
    
    # Store video IDs mapping for each window
    window_video_ids = []

    print("=" * 60)
    print(" LOADING AND MERGING DATASETS")
    print("=" * 60)

    for folder_name in subdirs:
        folder_path = export_dir / folder_name
        npz_path = folder_path / "keypoint_windows.npz"
        if not npz_path.exists():
            print(f"Warning: Subdirectory export not found at {npz_path}. Skipping.")
            continue
            
        print(f"Loading sub-dataset: {folder_name}...")
        npz_data = dict(np.load(npz_path, allow_pickle=True))
        
        kps = npz_data["keypoints"]  # (N, 60, 17, 2)
        bbs = npz_data["bboxes"]     # (N, 60, 4)
        scs = npz_data["keypoint_scores"] # (N, 60, 17)
        lbls = npz_data["labels"]    # (N,)
        
        n_win = len(lbls)
        print(f"  - Loaded {n_win} windows.")
        
        keypoints_list.append(kps)
        bboxes_list.append(bbs)
        scores_list.append(scs)
        labels_list.append(lbls)
        
        # 3. Track Clustering & Video ID Extraction (Per Sub-dataset)
        # Group windows into contiguous tracks strictly within this sub-dataset boundary
        local_groups = []
        current_group = [global_window_offset]
        for i in range(n_win - 1):
            w_curr = kps[i, 12:60]
            w_next = kps[i+1, 0:48]
            
            # Map index to the final merged array
            curr_idx = global_window_offset + i
            next_idx = global_window_offset + i + 1
            
            if np.allclose(w_curr, w_next, atol=1e-4):
                current_group.append(next_idx)
            else:
                local_groups.append(current_group)
                current_group = [next_idx]
        local_groups.append(current_group)
        
        print(f"  - Segmented sub-dataset into {len(local_groups)} trajectories.")
        
        # Assign synthetic video IDs to windows based on their trajectory groups
        local_video_ids = [None] * n_win
        for local_group_idx, g_indices in enumerate(local_groups):
            # Each trajectory group represents a unique track (person) from a unique video segment
            video_name = f"{folder_name}_video_{local_group_idx}"
            for idx in g_indices:
                local_idx = idx - global_window_offset
                local_video_ids[local_idx] = video_name
                
        # If the NPZ file contains real video IDs, override synthetic IDs (except for unknowns)
        if "video_ids" in npz_data:
            real_vids = npz_data["video_ids"]
            for i in range(n_win):
                vid = str(real_vids[i])
                if vid and vid != "unknown_video" and vid != "None":
                    local_video_ids[i] = vid
                    
        window_video_ids.extend(local_video_ids)
        global_window_offset += n_win

    if not labels_list:
        print("Error: No keypoint_windows.npz files loaded. Cannot proceed.")
        return

    # Concatenate all datasets
    raw_keypoints = np.concatenate(keypoints_list, axis=0)
    bboxes = np.concatenate(bboxes_list, axis=0)
    keypoint_scores = np.concatenate(scores_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    video_ids_array = np.array(window_video_ids, dtype=object)

    num_windows = len(labels)
    
    # Group window indices by resolved video ID to ensure leakage-free video-level splitting
    video_to_indices = {}
    for idx, vid in enumerate(window_video_ids):
        video_to_indices.setdefault(vid, []).append(idx)
    groups = list(video_to_indices.values())
    
    # Exclude 'track_gap_count' and 'valid_frame_ratio' from features list
    exclude_features = {"track_gap_count", "valid_frame_ratio"}
    columns = [col for col in feature_columns() if col not in exclude_features]
    
    print("\n" + "=" * 60)
    print(" COMBINED DATASET STATISTICS")
    print("=" * 60)
    print(f"Total merged windows        : {num_windows}")
    print(f"Total merged trajectories   : {len(groups)}")
    print(f"Target feature vector size  : {len(columns)} columns (track_gap_count & valid_frame_ratio excluded)")
    print("-" * 60)
    unique_classes, counts = np.unique(labels, return_counts=True)
    class_dist_str = ", ".join(f"Class {c}: {cnt}" for c, cnt in zip(unique_classes, counts))
    print(f"Combined class distribution : {class_dist_str}")
    print("=" * 60)

    # 2. Extract features using video_features.py
    print("\nExtracting Group A-E features for all windows...")
    X_engineered = np.zeros((num_windows, len(columns)), dtype=np.float32)
    
    null_or_gap_samples = []

    for i in range(num_windows):
        frames = []
        for f in range(60):
            frames.append({
                "bbox": bboxes[i, f].tolist(),
                "keypoints": raw_keypoints[i, f].tolist(),
                "keypoint_scores": keypoint_scores[i, f].tolist()
            })
        
        # Calculate raw window features
        features_dict = extract_window_features(frames)
        raw_features = features_dict["raw"]

        null_cols = []
        # Store in matrix in matching index order (excluding track_gap_count and valid_frame_ratio)
        for col_idx, col_name in enumerate(columns):
            val = raw_features.get(col_name)
            if val is None:
                null_cols.append(col_name)
                X_engineered[i, col_idx] = 0.0
            else:
                X_engineered[i, col_idx] = float(val)

        # Check for NULLs or other gap indicators if they ever arise
        gap_count = raw_features.get("track_gap_count", 0.0)
        if len(null_cols) > 0 or (gap_count is not None and gap_count > 0):
            null_or_gap_samples.append({
                "window_index": i,
                "null_columns": null_cols,
                "track_gap_count": gap_count if gap_count is not None else 0.0,
                "label": int(labels[i])
            })

        if (i + 1) % 400 == 0 or (i + 1) == num_windows:
            print(f"  - Extracted features for {i + 1}/{num_windows} windows...")

    # Post-process: Normalize/clamp hip_ankle vertical ratios to [0.0, 1.0] if they exceed 1.0
    print("\nNormalizing hip_ankle vertical ratios (capping values > 1.0 to 1.0)...")
    clamped_cols = 0
    for col_idx, col_name in enumerate(columns):
        if col_name == "hip_ankle_vertical_diff_mean" or col_name.startswith("step_hip_ankle_"):
            X_engineered[:, col_idx] = np.clip(X_engineered[:, col_idx], 0.0, 1.0)
            clamped_cols += 1
    print(f"  - Clamped values in {clamped_cols} hip_ankle columns to a maximum of 1.0.")

    # Print samples containing NULL values or track_gap_count > 0
    print("\n" + "=" * 60)
    print(" SAMPLES WITH NULL VALUES OR TRACK GAP COUNT > 0")
    print("=" * 60)
    print(f"Total such samples found: {len(null_or_gap_samples)} out of {num_windows} windows.")
    
    if len(null_or_gap_samples) > 0:
        print(f"\nListing first 15 samples:")
        print(f"{'Index':<8} | {'Label':<5} | {'Gap Count':<9} | {'NULL Cols count':<15} | {'NULL Columns'}")
        print("-" * 80)
        for s in null_or_gap_samples[:15]:
            null_cols_str = ", ".join(s["null_columns"]) if s["null_columns"] else "None"
            if len(null_cols_str) > 40:
                null_cols_str = null_cols_str[:37] + "..."
            print(f"{s['window_index']:<8} | {s['label']:<5} | {s['track_gap_count']:<9.1f} | {len(s['null_columns']):<15} | {null_cols_str}")
    print("=" * 60)

    # 4. Stratified Leakage-Free Splitting
    # For the dataset splitting process, the outputs (train, val, test) should have 
    # the same distribution of class samples as the overall dataset (approx 70/15/15 split).
    # We do a greedy stratified group assignment based on relative deficit normalization.
    
    # Calculate the class distribution of each trajectory group
    group_class_counts = []
    for g in groups:
        counts_dict = {0: 0, 1: 0, 2: 0}
        for idx in g:
            lbl = int(labels[idx])
            counts_dict[lbl] = counts_dict.get(lbl, 0) + 1
        group_class_counts.append(counts_dict)
        
    # Get total count per class
    unique_classes, total_counts = np.unique(labels, return_counts=True)
    class_totals = {int(c): int(count) for c, count in zip(unique_classes, total_counts)}
    
    # Target ratios and counts per split per class
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    targets = {
        split: {c: ratios[split] * class_totals.get(c, 0) for c in class_totals}
        for split in ratios
    }
    
    # Current accumulated counts per split per class
    current_counts = {
        split: {c: 0 for c in class_totals}
        for split in ratios
    }
    
    # Shuffle the group order with a fixed seed for reproducibility
    np.random.seed(42)
    group_indices = np.arange(len(groups))
    np.random.shuffle(group_indices)
    
    train_idx = []
    val_idx = []
    test_idx = []
    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    
    for g_idx in group_indices:
        g = groups[g_idx]
        counts = group_class_counts[g_idx]
        
        best_split = None
        best_score = -float("inf")
        
        # Decide split assignment by maximizing relative deficit dot product
        for split in ["train", "val", "test"]:
            score = 0.0
            for c in class_totals:
                count_in_group = counts.get(c, 0)
                if count_in_group > 0:
                    deficit = targets[split][c] - current_counts[split][c]
                    # Normalize deficit by global class totals to keep classes of different scales balanced
                    normalized_deficit = deficit / class_totals[c]
                    score += count_in_group * normalized_deficit
            
            if score > best_score:
                best_score = score
                best_split = split
                
        if best_split is None:
            best_split = "train"
            
        # Assign group to the chosen split
        split_indices[best_split].extend(g)
        for c in class_totals:
            current_counts[best_split][c] += counts.get(c, 0)

    # 5. Print Validation Samples
    print("\n" + "=" * 60)
    print(" SAMPLE FEATURE VALIDATION (First Window in Train split)")
    print("=" * 60)
    print(f"Train Label   : {labels[train_idx[0]]}")
    print(f"Train Video ID: {video_ids_array[train_idx[0]]}")
    print("First 10 pre-engineered feature values:")
    for col_idx in range(10):
        name = columns[col_idx]
        val = X_engineered[train_idx[0], col_idx]
        print(f"  - {name:<30} : {val:.6f}")
    print("=" * 60)

    # 6. Save splits into 3 distinct files
    np.savez_compressed(train_file, X=X_engineered[train_idx], y=labels[train_idx], video_ids=video_ids_array[train_idx])
    np.savez_compressed(val_file, X=X_engineered[val_idx], y=labels[val_idx], video_ids=video_ids_array[val_idx])
    np.savez_compressed(test_file, X=X_engineered[test_idx], y=labels[test_idx], video_ids=video_ids_array[test_idx])
    
    print("\nSuccessfully saved splits to:")
    print(f"  - Train: {train_file.absolute()}")
    print(f"  - Val  : {val_file.absolute()}")
    print(f"  - Test : {test_file.absolute()}")


if __name__ == "__main__":
    main()
