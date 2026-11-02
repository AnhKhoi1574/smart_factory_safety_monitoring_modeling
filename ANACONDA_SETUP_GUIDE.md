# Anaconda setup guide

Use Conda for the notebook workflows. Run these commands from the repository root
in a shell where `conda` is available: Anaconda Prompt/PowerShell on Windows, or
Bash/Zsh on Linux/macOS.

## 1. Create the general notebook environment

**All platforms:**

```sh
conda env create -f environment.yml
conda activate smart-factory-train
python -m ipykernel install --user --name smart-factory-train --display-name "Python (smart-factory-train)"
```

`environment.yml` declares Python 3.14 and the data, plotting and Behavior
modeling packages, plus pip-installed YOLO tools. It is an environment definition,
not a version lock for historical runs. `requirements.txt` contains the common
pip toolset; it is not a complete replacement for the Conda Behavior dependencies.

For PPE v2, also apply its supported Ultralytics range in the intended kernel:

**All platforms, repository root:**

```sh
python -m pip install -r ppe-detection/v2_pipeline/requirements.txt
```

Use a separate environment when preserving an existing training workstation.
Select the correct kernel in Jupyter before running cells.

## 2. Select PyTorch for the device

Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/)
for the target OS, Python and CPU/CUDA platform. Run its command inside the
activated environment. For a historical build, use the official
[previous-version instructions](https://pytorch.org/get-started/previous-versions/).

The selector is the primary installation reference; a CUDA command from another
workstation is not a hardware-independent setup instruction.

## 3. Check the selected interpreter

**All platforms, activated environment:**

```sh
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('build', torch.version.cuda)"
jupyter lab
```

## 4. Choose a workflow

- [PPE v2](ppe-detection/v2_pipeline/README.md): begin with Notebook 01; use the
  documented standard or direct route.
- [Sign](sign-detection/README.md): run the data and training notebooks in order.
- [Behavior](behavioral-detection/README.md): read the preprocessing workspace
  requirements, or begin modeling from supplied cleaned NPZ inputs.

Data placement and training instructions belong in those module guides. Model
setup for the applications is in [HANDOVER](docs/HANDOVER.md#application-model-paths).
