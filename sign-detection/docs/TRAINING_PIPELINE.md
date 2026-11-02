# Training Pipeline

Purpose: document candidate triage, ablation training, final training, and final test evaluation for the sign detector.

## Candidate Selection

Notebook 05 performs architecture triage before any ablation study. It trains
all configured lightweight YOLO candidates on `data_exp_D_full_pipeline.yaml`
so every architecture sees the same full-pipeline training condition:
original train data, offline augmented train data, and online augmentation
enabled through Ultralytics training arguments.

Candidate selection uses validation metrics only. Recall is the first sorting
priority because missed factory signs may fail to activate backend safety
rules. The test split is not evaluated or used for candidate selection in this
notebook. The top successful architecture from
`reports/training/candidate_model_ranking.csv` is the architecture Notebook 06
should use for ablation training.

## Ablation Study

Notebook 06 runs the ablation study after architecture selection is complete.
It resolves the selected architecture from
`reports/training/candidate_model_ranking.csv`, or falls back to
`selected_model` in `configs/training_config.yaml` when needed.

The selected architecture is fixed across all four experiments:
`exp_A_original_only`, `exp_B_online_aug`, `exp_C_offline_aug`, and
`exp_D_full_pipeline`. Online augmentation is enabled only for B and D through
training arguments. Validation metrics decide the best training configuration;
the test set is not used for ablation selection.

Notebook 07 should use the selected architecture and the top successful
ablation experiment for final training and final test evaluation.

## Final Evaluation

Notebook 07 performs final training and final test evaluation. It resolves the
selected architecture from Notebook 06 ablation results first, then falls back
to Notebook 05 candidate ranking or `configs/training_config.yaml`.

For final deployment, Notebook 07 intentionally selects
`exp_D_full_pipeline` even if another ablation experiment ranks higher on
validation metrics. This manual override is recorded in
`reports/training/final_training_summary.csv` because the full pipeline is
expected to be more robust in real factory CCTV conditions: it combines
offline CCTV-style augmentation with online training augmentation.

Notebook 07 is the first stage allowed to evaluate on the untouched test set.
The test metrics are for final reporting only and must not be used to revise
architecture, augmentation choices, confidence thresholds, or backend logic.

## Export

Final deployment exports are saved under `weights/final/`. ONNX is the default
export format. Export failures are recorded in
`reports/training/final_export_report.csv` and should not invalidate the final
test report.
