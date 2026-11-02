# Training repository handover

This handover covers the training repository, supplied model files, recorded
results and instructions for continuing the work. Source ownership is recorded in
[OWNERSHIP](../OWNERSHIP.md). Application operation is documented in the
[labelling tool](https://github.com/7718TMT/ppe_labeling/tree/inference) and
[dashboard](https://github.com/jwydth/PPE_demo/tree/develop).

## Receive the materials

| Material | How it is supplied |
|---|---|
| Repository | `https://github.com/AnhKhoi1574/smart_factory_safety_monitoring_modeling`; record the received branch and commit |
| Model package | `models.zip`, delivered outside Git |
| Datasets | Separate ZIPs supplied to De Heus by the team |

The local preparation directory is outside the training checkout. Transfer the
model package and dataset ZIPs through the company's agreed channel; they are not
part of a clone. Company repository access is arranged separately.

### Model package contents

```text
weights/
  ppe.pt
  sign.pt
  behavior.joblib
  pose.pt
  reid.pt
evidence/
  ppe/
  sign/
  behavior/
  configs/
  handover-checks/
```

The package uses the agreed filenames. Existing source artifacts are copied under
these names without modifying their contents. `pose.pt` and `reid.pt` support the
application's Behavior processing.

Unpack into a new receiving directory first. For a ZIP in the current directory:

**Windows PowerShell:**

```powershell
Expand-Archive -LiteralPath .\models.zip -DestinationPath .\training-models
Get-ChildItem -LiteralPath .\training-models\weights -File | Select-Object Name, Length
```

**Linux/macOS — Bash or Zsh:**

```sh
unzip models.zip -d training-models
ls -lh training-models/weights
```

Confirm all five named files exist and have content. Recorded training results
are summarized in [MODELS](MODELS.md).

## Place the datasets and continue

Follow these module guides when unpacking the separately supplied dataset ZIPs:

| Workstream | Placement and workflow |
|---|---|
| PPE | [Source lanes and frozen manifests](../ppe-detection/v2_pipeline/docs/DATASET_PREPARATION.md), then [training](../ppe-detection/v2_pipeline/docs/TRAINING_PIPELINE.md) |
| Sign | [YOLO inputs and split preparation](../sign-detection/docs/DATASET_PREPARATION.md), then [training](../sign-detection/docs/TRAINING_PIPELINE.md) |
| Behavior | [Preprocessing workspace or cleaned NPZ inputs](../behavioral-detection/README.md) |

Preserve image/label filenames and pairing. Keep received ZIPs and completed run
folders as original records. Use a new run/experiment ID for subsequent work.
Install/select the notebook environment using the [setup guide](../ANACONDA_SETUP_GUIDE.md).

Read each notebook's setup cell before execution. Running a training notebook can
start fitting or generate artifacts. Final evaluation controls and output paths
are documented once in the module's training guide.

## Application model paths

Copy the five model files into each application's weights directory:

- Labelling tool: `<labelling-repository>/weights/`.
- Dashboard: `<dashboard-repository>/backend/weights/`.

Use the labelling configuration guide and dashboard operations guide 
for `.env` setup and restart procedures. Use these relative path values with the
corresponding application's path resolver:

| Delivered file | Labelling-tool setting | Dashboard setting |
|---|---|---|
| `weights/ppe.pt` | `PPE_MODEL_PATH` | `MODEL_PATH` |
| `weights/sign.pt` | `SAFETY_SIGN_MODEL_PATH` | `SIGN_MODEL_PATH` |
| `weights/behavior.joblib` | `BEHAVIOR_MODEL_PATH` | `FALL_BEHAVIOR_MODEL_PATH` |
| `weights/pose.pt` | `POSE_MODEL_PATH` | `FALL_MODEL_PATH` |
| `weights/reid.pt` | Fixed repository-root worker path | `FALL_REID_MODEL_PATH` |

These are path values, so their `.env` syntax is the same across shells. Existing
installations should update their configured paths to these delivery filenames.
Application source/configuration files are not changed by this repository handover.

## Verification baseline

The source snapshot is branch `behavioral-detection`, commit
`9ce4863f8510a7a43e12873a927cfd4a4e910826`. Documentation and notebook Markdown are
working-tree changes over that commit. The starting working tree was clean.

Captured on **2026-11-01**, Windows 11 build 26200, AMD64. The main structural
summary was recorded at **14:06:24 Asia/Saigon**. Commands reused installed
environments; no dependencies were installed or changed.

| Check | Environment | Observed result |
|---|---|---|
| Python/notebook/YAML parsing | System Python 3.14.3, PyYAML 6.0.3 | 52 Python files, 21 notebooks and 15 YAML files passed |
| Local Markdown links | Same structural checker | 102 links checked across 16 Markdown files; passed |
| Notebook preservation | Comparison with source snapshot | Code, outputs, execution counts and metadata unchanged in both edited Behavior notebooks |
| Existing PPE tests | `smart-factory-safety`, Python 3.10.20 | 38 tests passed, none skipped; unittest reported 6.042 s |
| Compile all three source modules | Same Python 3.10.20 environment | Passed |
| Git whitespace check | `git diff HEAD --check` | Passed |
| Model package extraction | System Python 3.14.3 | Five non-empty model files extracted successfully |

The package contains **58 files** and is **66,362,379 bytes**. Captured summaries
and the repeatable structural/package scripts are included in
`evidence/handover-checks/`. The test log retains the suite's final result rather
than its full per-test console output.

| Extracted file | Size in bytes |
|---|---:|
| `ppe.pt` | 5,518,298 |
| `sign.pt` | 4,642,249 |
| `behavior.joblib` | 11,510,305 |
| `pose.pt` | 24,151,790 |
| `reid.pt` | 23,487,128 |

The checks cover source structure, fixture/mock tests and package presence.
Notebook outputs are retained from their recorded runs; no training or accuracy
evaluation was executed during this capture.

To repeat structural checks, run from the repository root with an environment
that provides the module dependencies. **All platforms, activated environment:**

```sh
python -m compileall -q ppe-detection/v2_pipeline/src sign-detection/src behavioral-detection/src
python -m unittest discover -s ppe-detection/v2_pipeline/tests -v
git diff HEAD --check
git status --short
```

Also parse YAML/notebook files and check local documentation links. Full notebook
execution, training, accuracy evaluation, GPU benchmarking and TensorRT export are
separate activities; they are not part of these structural checks.

## Receipt checklist

- [ ] Record the received repository branch and commit.
- [ ] Receive the separate dataset ZIPs and `models.zip`.
- [ ] Extract the package and confirm all five non-empty model files.
- [ ] Read [MODELS](MODELS.md) and the selected module's guide.
- [ ] Place datasets, select the environment and review notebook controls.
- [ ] Configure application paths if model deployment is required.
- [ ] Arrange company repository/data access with the sender.
