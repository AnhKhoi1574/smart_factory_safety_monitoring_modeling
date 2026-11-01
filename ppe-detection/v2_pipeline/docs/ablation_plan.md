# Ablation Plan

The ablation study starts only after candidate architecture selection is locked
using validation data.

## Controlled matrix

| Experiment | Dataset variant | Online augmentation |
| --- | --- | --- |
| `exp_A_original_only` | original | off |
| `exp_B_online_aug` | original | on |
| `exp_C_offline_aug` | original + offline | off |
| `exp_D_full_pipeline` | original + offline | on |

A/B resolve to the same `train_original.txt`. C/D resolve to the same
`train_original_plus_offline.txt`. All four share the same `val.txt`, model,
seed, epochs, image size, and non-augmentation training settings.

The matrix is declared in `configs/experiments.yaml`; it does not create four
dataset directories. One run-local dataset YAML is generated per experiment
and contains train/val only.

## Selection policy

Rank trained rows lexicographically by:

1. recall,
2. mAP50,
3. mAP50-95,
4. validation-image FPS,
5. smaller model size.

The selected experiment is written to the ablation run's `run.yaml`. Test data
is unavailable to Notebook 05 and cannot be used to revise the ranking.

Native ablation checkpoints are intermediate because Notebook 06 calibrates
resolution and Notebook 07 retrains the locked configuration. They are removed
after compact results are safely saved unless `keep_intermediates` is enabled.
