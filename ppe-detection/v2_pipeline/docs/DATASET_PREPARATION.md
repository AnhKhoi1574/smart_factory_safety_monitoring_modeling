# Dataset preparation

Paths in this guide are relative to `ppe-detection/v2_pipeline/`.

## Inputs

Place YOLO image/label pairs in these source lanes:

```text
data/input/open_source/{images,labels}/
data/input/factory_source/{images,labels}/
data/input/test_source/{images,labels}/
```

| Lane | Pipeline use |
|---|---|
| `open_source` | Training only |
| `factory_source` | Training and validation |
| `test_source` | Final test evaluation |

Use matching image and `.txt` label stems. Each non-empty label row is
`class_id x_center y_center width height`, with coordinates normalized to the
image. The class map is defined in
[`class_names.yaml`](../configs/class_names.yaml). Supported extensions, empty-label
policy, seed and split ratio are defined in
[`dataset_config.yaml`](../configs/dataset_config.yaml).

The labelling tool exports `images/` and `labels/` pairs. Unpack the supplied
training ZIPs into their designated lanes; preserve filenames and the images/labels
relationship. Delivery steps are in [HANDOVER](../../../docs/HANDOVER.md).

## 01 — Audit and profile

Notebook 01 checks image readability, pairs, YOLO row shape, class IDs, coordinate
bounds, positive box size, duplicate rows/stems and duplicate content. It profiles
class distribution without modifying source images or labels. Its run retains
`run.yaml`, validation CSVs and selected EDA figures. Resolve invalid inputs before
freezing splits.

## 02 — Freeze membership

Notebook 02 creates:

```text
artifacts/<split-run>/
  run.yaml
  manifests/
    split_assignments.csv
    train_original.txt
    val.txt
    test.txt
  csv/split_summary.csv
```

The default factory split is 80% train and 20% validation with seed 42. It
stratifies by PPE-class presence; singleton strata remain in train. All
`open_source` images join train, and `test_source` forms test.

The manifests reference original images by absolute path. Labels resolve through
the sibling `images/` to `labels/` layout; source images are not copied into
physical split/experiment folders. Preserve historical manifests as evidence.
For work on another machine, create a new split run from the locally placed inputs.

The split run records a test image-and-label content fingerprint. Final Notebooks
07/08 check it before their training/evaluation workflow.

## Generated files

Only offline augmentation creates new dataset images under
`data/generated/augmented/`. Keep original validation/test images unchanged.
Follow the [training pipeline](TRAINING_PIPELINE.md) after split preparation.
Generated data and artifacts remain outside Git.
