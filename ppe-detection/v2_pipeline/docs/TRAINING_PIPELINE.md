# Training pipeline

Paths are relative to `ppe-detection/v2_pipeline/`. First complete
[dataset preparation](DATASET_PREPARATION.md). Notebook links are in the
[module README](../README.md).

## Standard route

```text
01 audit → 02 freeze manifests → 03 offline augmentation QA
→ 04 architecture selection → 05 augmentation ablation
→ 06 resolution calibration → 07 final training and test
```

Notebook controls are in their setup cells. Training notebooks support dry runs;
06–08 default to `DRY_RUN=True`. Inspect setup cells before running the notebook.
A dry run can create a run artifact even though it does not train a model.

## 03 — Offline augmentation

Read only frozen `train_original.txt`. Generate mild, non-geometric IR/grayscale,
sunlight/glare and blur/JPEG simulations. Geometry is unchanged, so labels are
copied unchanged. Review the before/after figure and augmentation summary.

The combined `train_original_plus_offline.txt` includes original training paths
plus accepted outputs from this augmentation run. Validation/test are not augmented.
Existing augmentation targets require explicit regeneration to overwrite.

## 04–05 — Architecture and augmentation

[Training defaults](../configs/training_config.yaml) define candidate models,
epoch budgets, batch/image size and ranking priorities. Candidates share the same
training conditions so architecture is the variable. Failed/unsupported candidates
are reported separately.

The [experiment matrix](../configs/experiments.yaml) uses two train memberships:

| Experiment | Membership | Online augmentation |
|---|---|---|
| A | Original | Off |
| B | Original | On |
| C | Original + offline | Off |
| D | Original + offline | On |

Online parameters come from
[`augmentation_config.yaml`](../configs/augmentation_config.yaml), not persistent
image files. A/C neutralize the configured online transforms; B/D enable them.
A/B share the original manifest, and C/D share the combined manifest. All four use
the same validation membership and selected architecture.

Candidate/ablation ranking uses validation recall, mAP50, mAP50-95, FPS and model
size in the order defined by config. Notebook 05 records the selected experiment
in `run.yaml`. Notebooks 04–06 resolve train/validation only.

## 06 — Resolution calibration

Keep architecture and augmentation policy fixed and compare the configured
640/32, 960/16 and 1280/8 image-size/batch profiles on validation. Rank helmet
recall first, followed by helmet mAP50-95, overall mAP50-95 and FPS. Store the
selected profile and exact source run IDs in metadata for Notebook 07.

## 07 — Epoch calibration, refit and test

A real final run requires both `DRY_RUN=False` and
`ALLOW_TEST_EVALUATION=True`. Pin `SOURCE_RESOLUTION_RUN` when reproducing a chosen
experiment rather than relying on newest-run discovery.

1. Consume Notebook 06's selected model, experiment, image size and batch.
2. Train calibration on selected train membership and clean validation. Lock the
   first epoch with the best validation mAP50-95.
3. Restart from the original pretrained checkpoint. Refit selected train plus
   clean validation for the locked epoch count, with validation/early stopping
   disabled. Experiments C/D retain their offline training examples.
4. Retain refit `last.pt` as the final checkpoint, then evaluate test for reporting.
5. Save overall/per-class metrics, confusion matrix, bounded sample predictions
   and configured exports.

Training YAMLs omit test. The test-enabled evaluation YAML is created after refit.
The refit YAML uses the development manifest for both train/val keys to satisfy
Ultralytics' format, while runtime validation remains disabled. Final test metrics
do not feed back into model or threshold selection.

## 08 — Alternative direct route

After Notebook 02, Notebook 08 accepts an explicit `MODEL_NAME` and source/online
augmentation choices instead of running 03–07. Open-source data is optional;
offline augmentation is absent. `TRAIN_IMGSZ` and `TRAIN_BATCH` apply to both
calibration and refit. Calibration locks the epoch on train/validation, then a
fresh pretrained initialization is refitted on original train + validation before
final test reporting. It retains refit `last.pt`.

Use a new run for each intentional experiment and declare choices before viewing
test results.

## Outputs and retention

Each run writes under:

```text
artifacts/<timestamp>_<stage>_<descriptor>_s<seed>_<git-sha>/
  run.yaml
  manifests/
  figures/
  csv/
  models/
```

`run.yaml` records source run IDs, settings, runtime and retained outputs. Only
selected/final checkpoints and compact reports are retained by default. Native
Ultralytics `_work` and pretrained-cache directories are cleaned after retention
unless `keep_intermediates` is enabled. Treat completed artifacts as immutable.

A preprocessing run can use CPU; training defaults select device 0. Use an
appropriate device/worker count for a new run. The helpers shut down dataloader
workers between models. See [HANDOVER](../../../docs/HANDOVER.md#verification-baseline)
for structural checks and [MODELS](../../../docs/MODELS.md) for recorded results.
