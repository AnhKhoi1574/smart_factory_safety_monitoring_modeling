# Behavior classification

Classify `others`, `running` and `falling` from 141 transformed pose/motion
features. The notebooks cover feature preprocessing and tree/ensemble modeling;
the application supplies pose estimation, tracking and window feature extraction.

## Start here

1. Follow the repository [setup guide](../ANACONDA_SETUP_GUIDE.md).
2. Receive the separate data ZIP and place cleaned inputs as described below.
3. Read [behavior_preprocess.ipynb](behavior_preprocess.ipynb) when preparing
   features, or open [behavior_modeling.ipynb](behavior_modeling.ipynb) for supplied
   cleaned data.
4. See [MODELS](../docs/MODELS.md) for recorded results and
   [HANDOVER](../docs/HANDOVER.md) for delivery/setup.

## Preprocessing workspace

The preprocessing notebook runs in a feature-extraction workspace. Before
executing it, set the kernel working directory to the workspace that provides
`backend/app/services/video_features.py` and these precomputed inputs:

```text
train_split.npz
val_split.npz
test_split.npz
```

Each input contains `X` with 141 ordered features, integer `y`, and `video_ids`.
The notebook constructs column names from the external schema with two columns
omitted; the supplied matrix already has 141 columns. The official
[feature reference](docs/FEATURES_FOR_BEHAVIOR_DETECTION.md) defines the raw-to-model
mapping, formulas and exact order.

Choose `dataset_name` in the setup cell and run for each split. The notebook
applies training-only balancing and the documented feature transforms, then
exports `X`/`y` in `<split>_cleaned.npz`.

The [labelling export](https://github.com/7718TMT/ppe_labeling/blob/inference/docs/DATA_EXPORT.md)
contains pose windows and annotations. The preprocessing input described here is
a precomputed feature table. The [feature reference](docs/FEATURES_FOR_BEHAVIOR_DETECTION.md)
connects these representations to the model input contract.

## Model input contract

Place the preprocessing outputs here:

```text
behavioral-detection/data/train_cleaned.npz
behavioral-detection/data/val_cleaned.npz
behavioral-detection/data/test_cleaned.npz
```

Each NPZ supplies finite numeric `X` of shape `(N, 141)` and integer `y` of shape
`(N,)`. Labels are `0=others`, `1=running`, `2=falling`. The ordered feature names
are defined by [`build_feature_names()`](src/behavior_modeling/data.py). Keep this
order when preparing inference inputs.

The [ordered feature tables](docs/FEATURES_FOR_BEHAVIOR_DETECTION.md#ordered-model-feature-vector)
specify every column/group, transform, window setting and measurement unit.

## Modeling workflow

The notebook can locate this module from the repository/module directory. Select
the intended kernel and choose a new `EXPERIMENT_ID` before a complete run.
Running all cells starts tuning and fitting.

1. Load the fixed cleaned inputs and check their contract/distributions.
2. Tune seven tree/ensemble families, 50 Optuna trials per available family,
   using validation macro F1. XGBoost/LightGBM are optional candidates.
3. Compare five stability seeds: 17, 42, 83, 123 and 2026.
4. Treat candidates within 0.005 of the best mean validation macro F1 as tied;
   prefer lower seed variation, gap, complexity and inference time.
5. Choose a representative seed and compute validation permutation importance.
6. Refit the selected configuration on train + validation, then predict test for
   reporting. No test-informed tuning or feature selection follows.
7. Save the selected classifier and compact reports under a new experiment ID.

Reusable functions live in `src/behavior_modeling/{data,modeling,evaluation,artifacts}.py`.
The metadata records feature order, class mapping, parameters, seed, results and
library versions. Prediction CSV probability columns use the notebook's display
order; estimator probabilities are mapped through `model.classes_`.

## Outputs

```text
artifacts/<YYYYMMDD_model-family_purpose_vNN>/
  figures/   # five analytical figures
  csv/       # comparison, stability, importance, metrics and predictions
  model/     # selected classifier and metadata
```

Completed experiment directories are not overwritten. The saved training model is
`selected_behavior_classifier.joblib`; package naming and application model paths
are defined in [HANDOVER](../docs/HANDOVER.md#application-model-paths).

Use [FEATURES_FOR_BEHAVIOR_DETECTION](docs/FEATURES_FOR_BEHAVIOR_DETECTION.md) as the
official feature engineering reference. Structural checks are in the
[handover baseline](../docs/HANDOVER.md#verification-baseline).
