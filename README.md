# Smart Factory Safety Monitoring — Model Training

Prepare data, train and evaluate the PPE, safety-sign and worker-behavior models
for the RMIT capstone project with De Heus.

| Workstream | Model output | Start here |
|---|---|---|
| PPE | `people`, `helmets`, `vests`, `cleaning coverall` | [PPE v2](ppe-detection/v2_pipeline/README.md) |
| Safety signs | `helmet`, `vest`, `no thoroughfare`, `slippery` | [Sign detection](sign-detection/README.md) |
| Behavior | `others`, `running`, `falling` from pose/motion features | [Behavior classification](behavioral-detection/README.md) |

For the De Heus handover, start with [HANDOVER](docs/HANDOVER.md).
For delivered models and recorded results, read [MODELS](docs/MODELS.md).

## System boundaries

- The labelling tool creates, reviews and exports annotations and supports model inspection.
- This repository prepares training inputs, selects models and records evaluation.
- The dashboard uses supplied
  models for inference and monitoring.
  

## Set up and continue

1. Follow the [Anaconda setup guide](ANACONDA_SETUP_GUIDE.md).
2. Receive the separate dataset ZIPs and model package described in
   [HANDOVER](docs/HANDOVER.md).
3. Open the selected module's README and follow its notebook order.
4. Use validation for experiment decisions; reserve test evaluation for the
   reporting stage described by that pipeline.

Notebooks orchestrate the workflows; reusable logic lives in each module's
`src/`. [PPE v1](ppe-detection/v1_legacy/README.md) is historical work.

## Data and outputs

Datasets, generated data, experiment reports, YOLO runs and model weights are
excluded from Git. They are delivered separately. Preserve their directory layout
when unpacking. Keep source code, configs, notebooks and documentation in Git.

## Documentation

| Need | Document |
|---|---|
| Receive the repository and model package | [Handover](docs/HANDOVER.md) |
| Understand models and recorded results | [Models](docs/MODELS.md) |
| Set up Python/Jupyter | [Anaconda setup](ANACONDA_SETUP_GUIDE.md) |
| Prepare and train PPE | [PPE v2](ppe-detection/v2_pipeline/README.md) |
| Prepare and train Sign | [Sign](sign-detection/README.md) |
| Preprocess and model Behavior | [Behavior](behavioral-detection/README.md) |
| Understand Behavior feature engineering | [Feature reference](behavioral-detection/docs/FEATURES_FOR_BEHAVIOR_DETECTION.md) |
| Review ownership | [Ownership](OWNERSHIP.md) |

Documentation is maintained manually from implementation and recorded run outputs.
Archived material is historical and is not the operating guide.
