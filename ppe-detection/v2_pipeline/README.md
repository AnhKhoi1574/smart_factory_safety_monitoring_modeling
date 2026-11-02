# PPE detection v2

Train a four-class YOLO object detector. Class IDs/names are defined in
[`configs/class_names.yaml`](configs/class_names.yaml); violation logic runs in
the downstream application.

## Start here

1. Follow the repository [setup guide](../../ANACONDA_SETUP_GUIDE.md).
2. Place inputs using [dataset preparation](docs/DATASET_PREPARATION.md).
3. Follow the notebook order below. Select the standard or direct training route
   described in the [training pipeline](docs/TRAINING_PIPELINE.md).
4. Find recorded results and delivery names in [MODELS](../../docs/MODELS.md).

## Notebook order

| Notebook | Purpose |
|---|---|
| [01](notebooks/01_data_audit_and_eda.ipynb) | Audit inputs and profile the dataset |
| [02](notebooks/02_split_and_freeze_manifests.ipynb) | Freeze train/validation/test manifests |
| [03](notebooks/03_offline_augmentation_qa.ipynb) | Generate and review training augmentation |
| [04](notebooks/04_candidate_model_selection.ipynb) | Select architecture on validation |
| [05](notebooks/05_augmentation_ablation.ipynb) | Compare augmentation policies |
| [06](notebooks/06_small_object_resolution_calibration.ipynb) | Select input resolution and batch profile |
| [07](notebooks/07_final_training_and_test.ipynb) | Calibrate epochs, refit and evaluate |
| [08](notebooks/08_direct_model_training_and_test.ipynb) | Alternative direct model workflow after 02 |

Reusable helpers live in `src/`; settings live in `configs/`. Runs are stored
under ignored `artifacts/` directories. Inference/tracking helpers in this module
are placeholders; use the labelling tool or dashboard for application inference.

Structural verification commands are maintained in
[HANDOVER](../../docs/HANDOVER.md#verification-baseline).
