# Dataset preparation

Paths are relative to `sign-detection/`.

## Inputs

Place prepared YOLO pairs directly into:

```text
data/input/images/
data/input/labels/
```

Each image has a same-stem `.txt` label. Non-empty rows are
`class_id x_center y_center width height`, normalized to image dimensions.
Use the fixed mapping in [`class_names.yaml`](../configs/class_names.yaml).
Empty label files represent no-sign examples and are counted separately.

Unpack supplied image exports into these folders, preserving names and pairs. 
General receipt steps are in
[HANDOVER](../../docs/HANDOVER.md).

## 01 — Validate and profile inputs

Notebook 01 checks readability, pairs, classes, normalized coordinates, positive
box sizes and image boundaries. It profiles input class presence and no-sign
status without splitting or modifying inputs. Its principal outputs are:

```text
reports/validation/validation_report.csv
reports/profile/input_image_profile.csv
reports/profile/input_bbox_records.csv
reports/profile/input_summary.csv
```

## 02 — Split and inspect

Notebook 02 uses valid image profiles and the ratios/seed in
[`dataset_config.yaml`](../configs/dataset_config.yaml): 70% train, 20% validation,
10% test with seed 42 by default. It stratifies by image class presence/no-sign
status when the strata support splitting; otherwise it records a random-split
fallback.

Pairs are copied into `data/generated/splits_original/{train,val,test}/`, leaving
inputs unchanged. `data_splits_original.yaml` describes that layout for Ultralytics.
The notebook saves split reports in `reports/splits/`, post-split EDA in
`reports/eda/`, and selected figures in `reports/figures/`.

Review class/object counts, no-sign ratios, box size/position, resolution,
brightness/blur and duplicate checks before training. Keep validation/test fixed
for the subsequent experiments; use test for split inspection and final reporting.

## Generated data

Notebook 03 generates train-only augmentation. Notebook 04 builds A/B/C/D
experiment folders and the corresponding root `data_exp_*.yaml` files. Follow the
[training pipeline](TRAINING_PIPELINE.md). Data, reports, runs and weights are
outside Git.
