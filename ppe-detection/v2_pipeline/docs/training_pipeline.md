# Training Pipeline

## Sequence

```text
01 audit + EDA
-> 02 freeze manifests
-> 03 generate and review offline augmentation
-> 04 select architecture on validation
-> 05 select augmentation on validation
-> 06 calibrate small-object input resolution on validation
-> 07 final retrain + one untouched-test evaluation
```

Notebooks 04-06 cannot resolve a test manifest. Notebook 07 is the locked
conclusion and generates a dataset YAML with `test`. Optional Notebook 08 is a
separate direct-final path described below.

## Candidate selection

Candidate models use the configured `candidate_experiment` with identical
membership, augmentation, epochs, image size, batch, seed, and device settings.
Only architecture varies. Unsupported checkpoints are recorded and do not hide
successful rows.

Ultralytics is imported lazily. Automatic pretrained downloads are contained
inside the current artifact run rather than the notebook directory. Raw
stdout/stderr and Ultralytics logger output are suppressed. An
`on_fit_epoch_end` callback updates one Jupyter panel in place with the current
model, epoch, train losses, and validation metrics. Only failed runs retain a
bounded raw-log excerpt in the failures CSV.

The tracked runtime defaults use `batch: 32` and `workers: 6` for the intended
16 GB NVIDIA GPU and 20-core Windows workstation. Multiple dataloader workers
prevent image decoding and online augmentation from serializing on the Jupyter
kernel's main process. After every model, the training helper explicitly shuts
down both persistent loader pools, removes trainer/model references, runs
garbage collection, and clears unused CUDA allocator blocks before starting the
next candidate. If Windows worker spawning fails, reduce `workers` to 4 and
then 2. RAM caching is intentionally disabled
because Ultralytics warns that it may weaken deterministic execution; disk
caching is also avoided because it writes cache files beside source images.

## Small-object resolution calibration

Notebook 06 reads the model and augmentation experiment locked by Notebook 05
and compares exactly these tracked profiles:

| Profile | Image size | Batch |
| --- | ---: | ---: |
| `img640` | 640 | 32 |
| `img960` | 960 | 16 |
| `img1280` | 1280 | 8 |

All other settings and train/validation manifests remain fixed. Successful
`best.pt` checkpoints are re-evaluated on validation at their matching image
size and ranked by helmet recall, helmet mAP50-95, overall mAP50-95, then FPS.
No test manifest is copied or resolved. The selected profile, `imgsz`, and
`batch` are written to `run.yaml`; intermediate profile checkpoints are
removed by default because Notebook 07 retrains the selected configuration.

## Final training and test

Notebook 07 reads model, experiment, image size, and batch from a completed
Notebook 06 run. An
actual final run requires both:

```text
DRY_RUN = False
ALLOW_TEST_EVALUATION = True
```

`FINAL_EXPERIMENT_OVERRIDE` defaults to `None`. An override is accepted only
when it matches the experiment already calibrated by Notebook 06; otherwise
the notebook fails before test access and instructs the user to rerun
resolution calibration with that policy.

Before training or evaluation, it verifies the frozen test image-and-label
fingerprint. Every run also records Python/package versions, Git revision,
seed, CUDA version, and detected GPU names in `run.yaml`.

Final fitting has two phases:

1. Calibration starts from pretrained weights, trains on the selected A-D
   manifest, validates on clean frozen validation data, and locks the first
   epoch with maximum validation mAP50-95.
2. Refit restarts from the same pretrained checkpoint, combines the selected
   training manifest with validation, disables validation and early stopping,
   and runs exactly the locked epoch count. C/D therefore preserve offline
   augmented samples while adding clean validation images. Refit `last.pt` is
   the final checkpoint; calibration `best.pt` is not continued or deployed.

Both training YAMLs omit test. The test-enabled evaluation YAML is created
only after refit has completed.

The final run keeps:

- one descriptive refit checkpoint copied from `last.pt`,
- requested deployment export(s),
- overall and optional class-level test CSVs,
- one confusion matrix when available,
- one bounded prediction sample grid.

Calibration checkpoints, candidate/ablation checkpoints, native training
directories, and pretrained download caches are not retained by default.

## Optional direct train-plus-val workflow

Notebook 08 bypasses Notebook 03-07 selection outputs. It consumes Notebook
02's frozen `split_assignments.csv` and trains one explicit `MODEL_NAME`. The
supplied presets are `yolo12n.pt` and `yolo12s.pt`, while any
Ultralytics-supported checkpoint name or local model path may be entered.

`USE_OPEN_SOURCE_DATA=False` removes only `open_source` rows. All factory train
and validation rows remain. Offline augmentation is intentionally absent;
`USE_ONLINE_AUGMENTATION` applies the tracked Notebook 04 augmentation arguments
without writing more dataset images.

Notebook 08 uses two training phases for the same model and policy. Explicit
`TRAIN_IMGSZ` and `TRAIN_BATCH` settings apply identically to both phases:

1. Calibration trains on frozen `train`, validates on disjoint factory `val`,
   uses configured early stopping, and locks the first epoch with maximum
   validation mAP50-95. Its dataset YAML omits test.
2. Refit restarts from the original pretrained checkpoint, combines original
   `train + val`, disables validation and early stopping, and trains exactly
   the locked epoch count. It retains refit `last.pt`, not the calibration
   checkpoint.

Ultralytics requires a `val` key even when refit validation is disabled, so the
refit YAML explicitly points `train` and `val` to the same development
manifest. Refit progress displays losses only instead of unavailable metric
zeroes. The test manifest remains disjoint and is evaluated only after refit
when both real-run flags are authorized.

Notebook 08 is suitable for an intentional direct deployment run, not for
comparing multiple models or source-toggle alternatives on the test set. Each
such decision must be fixed before test metrics are viewed.

## Structural verification

Before any full training run:

```text
python -m compileall ppe-detection/v2_pipeline/src
python -m unittest discover -s ppe-detection/v2_pipeline/tests -v
```

Also validate notebook JSON/code cells and parse every YAML file. Full training
is not a structural test.
