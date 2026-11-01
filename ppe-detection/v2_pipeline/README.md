# PPE Detection v2 Pipeline

This directory contains the concise, reproducible training pipeline for the
Smart Factory Safety Monitoring System.

The detector predicts exactly four object classes:

```text
0 = person
1 = helmet
2 = vest
3 = cleaning_coverall
```

Violation and role logic is applied after detection. Do not train violation
states as detector classes.

## Data contract

Place immutable YOLO sources under:

```text
data/input/open_source/{images,labels}/   # train only
data/input/factory_source/{images,labels}/ # train + validation
data/input/test_source/{images,labels}/    # final untouched test only
```

Original images are referenced by text manifests; they are not copied into
physical split or experiment datasets. Only offline augmentation writes new
dataset images under `data/generated/augmented/`.

## Notebook order

1. `01_data_audit_and_eda.ipynb`
2. `02_split_and_freeze_manifests.ipynb`
3. `03_offline_augmentation_qa.ipynb`
4. `04_candidate_model_selection.ipynb`
5. `05_augmentation_ablation.ipynb`
6. `06_small_object_resolution_calibration.ipynb`
7. `07_final_training_and_test.ipynb`

Optional direct workflow:

8. `08_direct_model_training_and_test.ipynb`

Notebook 06 compares the reviewed 640/960/1280 input profiles on validation
only, with model and augmentation fixed. It ranks helmet recall first and
writes the selected `imgsz`/`batch` into run metadata for Notebook 07.

Notebook 07 first calibrates the best epoch on disjoint selected-train/val
membership. It then restarts from pretrained weights, refits selected train +
val for exactly that epoch count, retains refit `last.pt`, and evaluates test
once. Experiments C/D keep their offline-generated training samples during
refit; validation images are added as clean development examples.

Notebook 08 bypasses candidate and augmentation selection. It calibrates one
explicitly named model on frozen `train`/`val`, locks the best validation epoch,
then restarts from pretrained weights and refits original `train + val` for
that epoch count before evaluating the independent test set. Open-source data
is optional. It is an alternative to Notebooks 03-07, not an additional
model-selection stage.

All training notebooks expose a dry-run option. Notebooks 06-08 default to dry
run, and final test evaluation additionally requires explicit authorization.
There is no inference or tracking notebook in this pipeline.

## Experiment model

Four ablations use only two physical training memberships:

| Experiment | Training manifest | Online augmentation |
| --- | --- | --- |
| A | original | off |
| B | original | on |
| C | original + offline | off |
| D | original + offline | on |

The matrix is declared in `configs/experiments.yaml`. Each run generates its
own Ultralytics dataset YAML inside the run artifact; there are no root
`data_exp_*.yaml` files.

## Artifacts

Every notebook execution creates one immutable directory:

```text
artifacts/<timestamp>_<stage>_<descriptor>_s<seed>_<git-sha>/
  run.yaml
  manifests/
  figures/
  csv/
  models/
```

Keep outputs compact: one summary CSV, only decision-relevant figures, and only
selected/final checkpoints. Native Ultralytics work directories are temporary
and are cleaned after successful retention unless configured otherwise.

## Test isolation

- Candidate, ablation, and resolution-calibration YAMLs contain only train and val.
- Only Notebook 07 and the optional direct Notebook 08 receive `test.txt`.
- Both final workflows verify the frozen test image-and-label content
  fingerprint before training or evaluation.
- The final test result is reporting-only and must not reopen architecture,
  augmentation, hyperparameter, or confidence-threshold selection.
- An actual final run requires `ALLOW_TEST_EVALUATION=True` and creates a new
  immutable run ID.

## Install and verify

Install the PPE-specific dependencies in the intended Conda/Jupyter kernel:

```text
pip install -r ppe-detection/v2_pipeline/requirements.txt
```

Fast structural checks do not train models:

```text
python -m compileall ppe-detection/v2_pipeline/src
python -m unittest discover -s ppe-detection/v2_pipeline/tests -v
```

Reusable logic lives in `src/`; notebooks contain focused orchestration,
bounded outputs, and observations written immediately after results.
