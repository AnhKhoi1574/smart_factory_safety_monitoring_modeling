# Training pipeline

Paths are relative to `sign-detection/`. Complete
[dataset preparation](DATASET_PREPARATION.md) first. Notebook links are in the
[module README](../README.md).

## 03 — Offline augmentation

Generate from `data/generated/splits_original/train/` only. Validation/test remain
original. Parameters and generation ratios live in
[`augmentation_config.yaml`](../configs/augmentation_config.yaml).

- Geometric transforms simulate rotation, scale, translation and viewpoint
  changes. Transform box corners and clip the new boxes to the image.
- Photometric transforms simulate grayscale/IR, light, shadow and contrast;
  labels remain unchanged.
- Weather/quality transforms simulate blur, compression, noise, dirty lens and
  low-resolution effects; labels remain unchanged.
- Synthetic placement is optional and disabled by default; pasted signs require
  new matching boxes when enabled.

No-sign samples keep empty labels after augmentation. Review the saved summaries
and visual examples before building experiments.

## 04 — Experiment datasets

Build four directories under `data/generated/experiments/`:

| Experiment | Training data | Online augmentation |
|---|---|---|
| `exp_A_original_only` | Original | Off |
| `exp_B_online_aug` | Original | On |
| `exp_C_offline_aug` | Original + offline | Off/minimal |
| `exp_D_full_pipeline` | Original + offline | On |

Validation/test are copied identically into all four directories. Notebook 04
builds datasets and YAMLs; online augmentation is applied later as training args.

## 05–06 — Model and augmentation comparison

[Training config](../configs/training_config.yaml) declares candidate checkpoints,
epoch budgets, seed, image size, batch, device and ranking priorities. Review the
notebook setup flags, including `DRY_RUN`, before execution.

Notebook 05 compares candidates using `data_exp_D_full_pipeline.yaml` under the
same training condition. It ranks validation recall, mAP50, mAP50-95, FPS and model
size and writes `reports/training/candidate_model_ranking.csv`.

Notebook 06 selects that architecture and compares A/B/C/D on validation. It writes
`reports/training/ablation_ranking.csv`. Test results do not participate in either
comparison. Unsupported models or failed runs are reported separately.

## 07 — Final training, evaluation and export

Notebook 07 resolves the architecture from ablation ranking, then candidate
ranking, then explicit config. The final setup cell specifies the experiment used
for fitting; the delivered training record uses `exp_D_full_pipeline` with online
augmentation enabled.

Review the notebook's training/evaluation/export controls before running cells.
It trains the final model, retains/copies `best.pt` and `last.pt` into
`weights/final/`, evaluates `best.pt` on test for reporting, and can save sample
predictions and deployment exports. ONNX is configured by default.

Reports are written under `reports/training/`:

```text
final_training_summary.csv
final_test_results.csv
final_class_metrics.csv
final_export_report.csv
```

The summary records model/experiment choices and output paths. Use validation for
training decisions; final test metrics are reporting outputs. Preserve completed
reports before starting a new experiment in the same workspace.

Application class-to-rule behavior is maintained in the dashboard. Notebook 08
and inference helper files in this module are placeholders. See
[MODELS](../../docs/MODELS.md) for recorded results and
[HANDOVER](../../docs/HANDOVER.md#verification-baseline) for structural checks.
