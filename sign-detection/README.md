# Factory sign detection

Train a YOLO detector for four factory safety-sign classes. The fixed mapping is
defined in [`configs/class_names.yaml`](configs/class_names.yaml). Application
code uses the detections to activate safety rules or propose zones.

## Start here

1. Follow the repository [setup guide](../ANACONDA_SETUP_GUIDE.md).
2. Place YOLO images/labels using [dataset preparation](docs/DATASET_PREPARATION.md).
3. Run the notebooks below using the [training pipeline](docs/TRAINING_PIPELINE.md).
4. See [MODELS](../docs/MODELS.md) for recorded results and delivery filenames.

| Notebook | Purpose |
|---|---|
| [01](notebooks/01_validate_and_profile_input_dataset.ipynb) | Validate and profile inputs |
| [02](notebooks/02_split_and_verify_dataset.ipynb) | Create and inspect original splits |
| [03](notebooks/03_offline_augmentation.ipynb) | Generate training augmentation |
| [04](notebooks/04_build_ablation_datasets.ipynb) | Build A/B/C/D experiment datasets |
| [05](notebooks/05_candidate_model_training.ipynb) | Compare candidate architectures |
| [06](notebooks/06_ablation_study.ipynb) | Compare augmentation experiments |
| [07](notebooks/07_final_training_and_evaluation.ipynb) | Final training, test reporting and export |

Notebook 08 and `src/inference/` are placeholders, not a runnable application
inference workflow. Use the labelling tool or dashboard for that workflow.

Reusable helpers live in `src/`; paths, augmentation and training defaults live
in `configs/`. Generated data, reports, runs and weights are local outputs outside
Git. Receipt and verification instructions are in [HANDOVER](../docs/HANDOVER.md).
