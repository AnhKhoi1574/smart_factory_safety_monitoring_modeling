# Dataset Preparation

## Source lanes

The pipeline reads three immutable YOLO source lanes:

```text
data/input/open_source/{images,labels}/
data/input/factory_source/{images,labels}/
data/input/test_source/{images,labels}/
```

- Open-source samples are train-only.
- Factory samples are split into train and validation.
- Test-source samples form the independent final test set.

Each image uses a matching sibling `.txt` label with the same stem. The fixed
class map is stored only in `configs/class_names.yaml`.

## Audit

Notebook 01 validates image readability, image-label pairing, supported image
types, five-value YOLO rows, numeric normalized coordinates, positive box size,
box boundaries, class IDs, duplicate annotation rows, duplicate stems, and
duplicate content. Empty labels follow `allow_empty_labels` in
`configs/dataset_config.yaml`.

The audit is read-only. Invalid rows block split freezing.

## Manifest split

Notebook 02 creates small run-local files instead of copying images:

```text
manifests/train_original.txt
manifests/val.txt
manifests/test.txt
manifests/split_assignments.csv
```

Factory train/validation membership is deterministic for the configured seed
and grouped by helmet, vest, and cleaning-coverall presence. Singleton groups
remain in train and produce a warning.

All manifests contain absolute image paths. Ultralytics resolves labels through
the sibling `images/` -> `labels/` layout. Train, validation, and test path
overlap is rejected.

Notebook 02 also fingerprints the ordered test image and label bytes. Notebook
06 recomputes that SHA-256 value before final training/evaluation and stops if
the supposedly untouched test source changed.

## Storage policy

Do not create active `data/generated/splits/` or
`data/generated/experiments/` trees. Existing legacy generated folders may
remain locally until a separately authorized cleanup after parity checks.
