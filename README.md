# Investigating Attention Mechanisms in Graph Neural Networks for Graph Classification

EPSRC summer research internship at the University of Leicester.

Researcher: Aneesh  
Supervisor: Dr Furqan Aziz



## Purpose

Investigate graph attention design using GAT and GATv2, with GCN,
GraphSAGE and GIN as contextual baselines, on MUTAG, PROTEINS and NCI1.



## Development status

Implementation through Stage 6.4 is present. Dataset and model inspections,
development fits and the epoch-budget audit are recorded in log/ and results/.
The cross-validation protocol is implemented; the Stage 6.5 reference fits
have not yet been run in this snapshot.



## Code and records

- src/ contains dataset loading and partitioning, model definitions, training,
  evaluation and result recording.
- experiments/train_cv.py owns the current shared settings and runs the
  reference cross-validation matrix. It was previously named run_cv.py.
- experiments/ablation.py defines the attention variants and reuses the same
  settings and cross-validation function. Running this module directly only
  prints the variant definitions and example result paths.
- experiments/inspections/ contains methodological inspections, with dataset
  and model inspections in its datasets/ and models/ subdirectories.
- archive/src/ and archive/experiments/ preserve superseded implementations,
  including the original development train.py and two-value evaluator. These
  are historical source records; active code does not import them.
- log/ records development observations, decisions and actual commands.
- results/ contains the saved evidence and selected model states.



## Cross-validation protocol

The reference matrix contains GCN, GraphSAGE, GIN, GAT and GATv2 on MUTAG,
PROTEINS and NCI1. Five stratified outer folds and one fresh fit per fold give
75 reference fits. Each fold uses approximately 70% of the dataset for
fitting, 10% for validation and 20% for outer-test assessment. The same
partitions are reused across models and variants.

Every valid fit runs for 500 epochs with batch size 32 and Adam, using a
learning rate of 0.01 and weight decay of 0.0005. The state with the lowest
validation cross-entropy is restored before one outer-test assessment;
the earliest epoch wins an exact tie. Non-finite training, validation or
outer-test loss raises an error.

The 500-epoch budget was informed by a 27-trajectory development-validation
audit on the same benchmark datasets later used for CV. It was fixed before
any CV results existed, without using development-test scores. This prior
use of the benchmark data is part of the methodology and limits claims of
assessment independence. log/log6.md records the clean rerun, the earlier
interrupted run and the budget decision.

The additional head-count and uniform-attention fits bring the full planned
matrix to 135 fits. Comparisons reuse the relevant reference results;
the fitted-attention analysis and intervention reuse selected GAT states.



## Running the code

Run modules from the repository root in the configured environment.
The existing partition and ablation inspections do not train models:

```powershell
python -m experiments.inspections.inspect_cv_splits
python -m experiments.ablation
```

After committing the reviewed source and settings, the following command
starts the 75 reference fits. It is a training command, not an inspection:

```powershell
python -m experiments.train_cv
```

Each fold saves a paired JSON record and validation-selected .pt state.
Existing result or state paths are rejected before fitting that model/dataset
group. Completed groups must be excluded explicitly when continuing a
partially completed matrix. Record source and settings before fitting,
then record results and logs afterwards.



## Environment setup

Verified with Python 3.14.7 on Windows 11, using an NVIDIA GeForce
RTX 4070 Laptop GPU and NVIDIA driver 616.64.

Install Python 3.14, 64-bit, if it is not already available.
Python must be installed before creating the virtual environment.

From the project root, run these commands in PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

requirements.txt records the verified package versions and the additional
package index needed for the CUDA 13.0 PyTorch build.

The virtual environment and downloaded datasets are excluded from Git.
Environment verification is recorded in log/log0.md.