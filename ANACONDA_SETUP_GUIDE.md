# Anaconda Setup Guide

This project is designed around an Anaconda workflow for Jupyter-heavy PPE detector development. The environment file installs the general data, visualization, and YOLO tooling on Python 3.14. PyTorch GPU support is installed as a follow-up step so each can choose the right build for their machine.

## 1. Create the Base Environment

From the repository root:

```powershell
cd C:\Github\smart-factory-safety-monitoring-system
conda env create -f environment.yml
conda activate smart-factory-train
python -m ipykernel install --user --name=smart-factory-train --display-name "Python (smart-factory-train)"
```

## 2. Install PyTorch for Your Device

The `environment.yml` intentionally does not pin `torch` so you can install a CPU or CUDA build that matches your workstation.

### Option A: NVIDIA GPU, CUDA 13.2 wheels

Use this if your NVIDIA environment is set up for CUDA 13.2 and you want the matching PyTorch wheel index:

```powershell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

### Option B: NVIDIA GPU, CUDA 13.0 wheels

Use this if your NVIDIA environment is standardized on CUDA 13.0-compatible drivers:

```powershell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Option C: NVIDIA GPU, CUDA 12.6 wheels

Use this if your NVIDIA environment is standardized on CUDA 12.6-compatible drivers:

```powershell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Option D: CPU-only fallback

```powershell
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

These install options are based on the official PyTorch installation guidance:

- https://docs.pytorch.org/get-started/locally/
- https://docs.pytorch.org/get-started/previous-versions/

## 3. Verify CUDA Availability

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Optional system-level check:

```powershell
nvidia-smi
```

## 4. Open the v2 Pipeline

After the environment is ready, work from the notebooks in:

- `ppe-detection/v2_pipeline/notebooks/01_validate_and_merge_dataset.ipynb`
- through
- `ppe-detection/v2_pipeline/notebooks/09_inference_tracking_demo.ipynb`

The notebooks import reusable helpers from `ppe-detection/v2_pipeline/src/`.

## 5. Notes for Teammates

- Each teammate should contribute source folders in the form `images/` plus `labels/`.
- Do not place large datasets, generated YOLO runs, or trained weights under Git-tracked paths outside the ignored artifact folders.
- The legacy v1 assets remain available under `ppe-detection/v1_legacy/` for reference only.
