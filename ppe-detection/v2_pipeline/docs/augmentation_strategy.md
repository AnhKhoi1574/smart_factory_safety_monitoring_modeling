# Augmentation Strategy

## Two layers

The pipeline separates:

- Offline augmentation: creates real train-only image files.
- Online augmentation: Ultralytics training arguments with no persistent
  dataset output.

Always freeze original train/validation/test membership before augmentation.
Validation and test images are never augmented.

## Offline augmentation

Notebook 03 reads only `train_original.txt` and generates mild, non-geometric:

- IR/grayscale CCTV simulation,
- sunlight/glare simulation,
- blur/JPEG-compression simulation.

Because geometry does not change, labels are copied byte-for-byte. Existing
targets are not overwritten unless regeneration is explicitly enabled.

The notebook retains:

- `csv/offline_augmentation_summary.csv`,
- `figures/offline_augmentation_before_after.png`,
- `manifests/train_original_plus_offline.txt`.

The combined manifest is built from original paths plus only outputs accepted
from the current augmentation run, preventing stale files from entering an
experiment silently.

## Online augmentation

Configured online values live in `configs/augmentation_config.yaml`. Experiment
A/C receives neutral values for every configured key; B/D receives the tracked
values. Online settings never change the dataset YAML.

If geometric offline transforms are added later, labels must be transformed
with the images and the visual QA contract must be updated.
