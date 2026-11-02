# Models and recorded results

This page summarizes the model delivery and results already recorded by the
training workflows. Values below are rounded from saved outputs; this handover
does not run a new accuracy evaluation. Dataset and model receipt instructions
are maintained in [HANDOVER](HANDOVER.md).

## Delivered files

The model ZIP uses these application-facing names:

| File | Purpose | Model family / input |
|---|---|---|
| `weights/ppe.pt` | Detect people and protective equipment | YOLO12n; image input; four object classes |
| `weights/sign.pt` | Detect configured safety signs | YOLOv9t; image input; four sign classes |
| `weights/behavior.joblib` | Classify worker behavior | ExtraTrees; 141 transformed features |
| `weights/pose.pt` | Produce worker skeletons | YOLO26s-pose; 17 COCO keypoints |
| `weights/reid.pt` | Appearance features for tracking | YOLO11m-cls backbone used for embeddings |

PPE, Sign and Behavior are the project's modeling workstreams. Pose and ReID are
supplied inference dependencies. Application tracking, zones, thresholds and
incident logic are maintained in the labelling tool/dashboard.

## PPE

Class order is `0=person`, `1=helmet`, `2=vest`, `3=cleaning_coverall`, as defined
in the [class config](../ppe-detection/v2_pipeline/configs/class_names.yaml).
Objects are associated with people by the application; the detector does not
predict violation classes.

### Recorded final run

Run: `20260826T062011Z_direct-final_yolo12n-factory-only_s42_027537d`.

- Initialization: `yolo12n.pt`; 100 epochs, image size 640, batch 32, seed 42.
- Development membership: original factory-lane train + validation, 859 images.
- Online augmentation enabled; open-source/offline augmentation disabled.
- Test membership: 117 images.
- Saved checkpoint: `models/yolo12n_train-plus-val.pt`; delivered as `ppe.pt`.

| Test metric | Value |
|---|---:|
| Precision | 0.8914 |
| Recall | 0.7929 |
| mAP50 | 0.8292 |
| mAP50-95 | 0.5366 |

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| person | 0.9662 | 0.8785 | 0.9292 | 0.6511 |
| helmet | 0.8681 | 0.5670 | 0.6264 | 0.4085 |
| vest | 0.8943 | 0.7596 | 0.8237 | 0.5317 |
| cleaning_coverall | 0.8367 | 0.9667 | 0.9377 | 0.5549 |

Source: the run's `run.yaml`, `csv/direct_test_overall_metrics.csv` and
`csv/direct_test_class_metrics.csv`, supplied under `evidence/ppe/<run-id>/` in
the ZIP. Split preparation records are under `evidence/ppe/<split-run-id>/`.
The operating workflow is in the [PPE training guide](../ppe-detection/v2_pipeline/docs/TRAINING_PIPELINE.md).

## Sign

Class order is `0=M014_Helmet`, `1=M015_Vest`, `2=P004_NoThoroughfare`,
`3=W011_Slippery`, defined in the
[class config](../sign-detection/configs/class_names.yaml).

### Recorded final training and test

- Model: `yolov9t.pt`, experiment `exp_D_full_pipeline`.
- 100 epochs, image size 640, batch 16, seed 42.
- Training combines original and offline-augmented images with online augmentation.
- Final checkpoint: `sign-detection/weights/final/best.pt`; delivered as `sign.pt`.
- Recorded test: 47 images, 48 annotated objects.

| Test metric | Value |
|---|---:|
| Precision | 0.9341 |
| Recall | 0.9464 |
| mAP50 | 0.9831 |
| mAP50-95 | 0.7974 |

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| M014_Helmet | 0.8773 | 1.0000 | 0.9888 | 0.8278 |
| M015_Vest | 1.0000 | 0.8854 | 0.9757 | 0.7696 |
| P004_NoThoroughfare | 0.8589 | 1.0000 | 0.9950 | 0.8412 |
| W011_Slippery | 1.0000 | 0.9002 | 0.9729 | 0.7509 |

Source: the saved training/test outputs in
[Notebook 07](../sign-detection/notebooks/07_final_training_and_evaluation.ipynb),
also included under `evidence/sign/`. See the
[Sign training guide](../sign-detection/docs/TRAINING_PIPELINE.md) for the pipeline.

## Behavior

The training feature contract is documented in
[FEATURES_FOR_BEHAVIOR_DETECTION](../behavioral-detection/docs/FEATURES_FOR_BEHAVIOR_DETECTION.md).
Labels are `0=others`,
`1=running`, `2=falling`. The classifier consumes 141 transformed pose/motion
features over a 60-frame window at canonical 24 FPS.

### Recorded modeling experiment

Experiment: `20260825_tree_ensemble_refactor_v01`.

- Seven tree/ensemble families; 50 Optuna trials per available family.
- Validation macro F1 is the tuning objective; five seeds measure stability.
- Selected model: Extra Trees, seed 17, 300 estimators, maximum depth 23.
- Cleaned inputs: 3,600 train rows, 786 validation rows and 763 test rows.
- Final refit uses 4,386 development rows.
- Mean validation macro F1: **0.9208**, seed standard deviation **0.0036**.

| Test metric | Value |
|---|---:|
| Accuracy | 0.9502 |
| Macro precision | 0.9394 |
| Macro recall | 0.9291 |
| Macro F1 | 0.9341 |
| Weighted F1 | 0.9501 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| others | 0.9527 | 0.9706 | 0.9616 | 477 |
| running | 0.9012 | 0.8902 | 0.8957 | 82 |
| falling | 0.9643 | 0.9265 | 0.9450 | 204 |

Source: this experiment's `selected_behavior_classifier_metadata.json`,
`validation_seed_stability_summary.csv` and `final_test_classification_metrics.csv`,
supplied under `evidence/behavior/<experiment-id>/`. The
[modeling notebook](../behavioral-detection/behavior_modeling.ipynb) contains the
saved experiment outputs. Results in this section describe that recorded run.

## From data to application

| Workstream | Data and preparation | Training/evaluation outputs | Delivery |
|---|---|---|---|
| PPE | YOLO pairs → source lanes → frozen manifests → train augmentation | Run metadata, selected checkpoint, overall/per-class CSVs | `ppe.pt` |
| Sign | YOLO pairs → original split → A/B/C/D datasets | Notebook outputs, final checkpoint and exports | `sign.pt` |
| Behavior | Pose/window features → transformed cleaned NPZ → modeling | Experiment metadata, classifier, metrics and figures | `behavior.joblib` |

Hand the model ZIP to the application maintainer and configure paths using
[application model setup](HANDOVER.md#application-model-paths). Existing ONNX
exports remain training outputs; this package supplies the portable `.pt` and
`.joblib` files. TensorRT build/deployment belongs to the dashboard workflow.
