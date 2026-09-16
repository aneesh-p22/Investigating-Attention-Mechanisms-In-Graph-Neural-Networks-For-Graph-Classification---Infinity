# Investigating Attention Mechanisms in Graph Neural Networks for Graph Classification

EPSRC summer research internship at the University of Leicester.

Researcher: Aneesh  
Supervisor: Dr Furqan Aziz



## Purpose

Investigate graph attention design using GAT and GATv2, with GCN,
GraphSAGE and GIN as contextual baselines, on MUTAG, PROTEINS and NCI1.

The four research questions concern fixed-width attention-head count,
matched GAT and GATv2 models, uniform-attention retraining, and fitted
attention characterisation with a uniform intervention without retraining.



## Development status

The experimental implementation and all four attention investigations are
complete. The saved evidence contains 135 cross-validation fits, each with
a paired JSON record and validation-selected model state in results/:

- 75 reference fits across five models and three datasets.
- 45 additional GAT fits with one, two or four first-layer heads.
- 15 uniform-attention GAT fits.

The eight-head GAT and GATv2 reference results are reused in the relevant
comparisons. Fitted-attention analysis and intervention reuse the 15 selected
reference GAT states without additional optimisation. The analysis covers
every graph once in its corresponding outer-test fold and is preserved in
results/rq4_fitted_gat_attention.json.

The report directory currently contains the LaTeX skeleton and bibliography;
substantive report writing remains to be completed.



## Code and records

- src/ contains dataset loading and partitioning, model definitions, training,
  evaluation and result recording.
- experiments/train_cv.py owns the shared assessment settings and runs the
  reference cross-validation matrix.
- experiments/result_validation.py provides the shared checks for saved
  cross-validation groups, partitions, predictions and runtime conditions.
- experiments/summarise.py validates the reference, head-count and uniform
  groups, then reports the reference and RQ1 through RQ3 summaries. These
  include all five fold accuracies, their unweighted mean and sample standard
  deviation, paired accuracy differences, parameter counts and training times.
- experiments/ablation.py defines the attention variants and reuses the same
  settings and cross-validation function. Running this module directly only
  prints the variant definitions and example result paths.
- experiments/runners/train_heads.py runs the additional head-count fits.
- experiments/runners/train_uniform.py runs the uniform-attention fits.
- experiments/analyse_fitted_gat.py performs RQ4 attention characterisation
  and intervention using the selected reference GAT states. It saves graph
  records, fold results and dataset summaries in one JSON file.
- experiments/inspections/ contains methodological inspections, with dataset
  and model inspections in its datasets/ and models/ subdirectories. The
  epoch-budget inspection retains its historical development-only settings.
- archive/src/ and archive/experiments/ preserve superseded implementations,
  including the original development train.py and two-value evaluator. These
  are historical source records; active code does not import them.
- results/ contains the saved evidence and selected model states.
- report/ contains the LaTeX source and bibliography.



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
assessment independence.

The additional head-count and uniform-attention fits bring the completed
matrix to 135 fits. RQ1 varies first-layer heads through 1, 2, 4 and 8 at a
fixed total width of 64. RQ2 compares GAT with general-form GATv2 using
share_weights=False in both layers. RQ3 trains fresh GAT models with the
attention-scoring tensors zeroed and frozen in both layers.

RQ4 characterises the fitted reference GAT coefficients in each layer using
one minus normalised attention entropy. Receivers with only one incoming
entry are excluded from entropy and retained as counts. The intervention
zeros the attention-scoring tensors in both layers while preserving the
remaining fitted parameters. It records changes in cross-entropy and
accuracy, defined as intervention minus learned, and the prediction-flip
rate. This measures fitted-model behaviour and sensitivity, without
establishing explanation faithfulness or domain causality.

Accuracy summaries use the arithmetic mean and sample standard deviation
across five outer folds. Accuracy differences are expressed in percentage
points. Training time covers training passes only, including data loading,
device transfers and optimisation, and excludes validation, test evaluation
and result saving. It is distinct from complete program elapsed time.



## Running the code

Run modules from the repository root in the configured environment.
Summarise the saved reference results and RQ1 through RQ3 with:

```powershell
python -m experiments.summarise
```

This reads the existing records and validates them against the loaded
datasets. Dataset files are downloaded on first use if they are not already
available. It does not train models or repeat model inference.

RQ4 already has saved summaries in results/rq4_fitted_gat_attention.json.
To print them using the existing table formatter, start Python from the
repository root:

```powershell
python
```

Then enter:

```python
import json
from experiments.analyse_fitted_gat import print_summary

with open("results/rq4_fitted_gat_attention.json", encoding="utf-8") as result_file:
    result = json.load(result_file)

print_summary(result["datasets"])
exit()
```

This prints the saved fold values, means, sample standard deviations and
receiver counts without repeating attention extraction or model inference.

The partition and ablation inspections do not train models:

```powershell
python -m experiments.inspections.inspect_cv_splits
python -m experiments.ablation
```

The following commands produce the 75 reference fits, 45 additional
head-count fits and 15 uniform-attention fits respectively:

```powershell
python -m experiments.train_cv
python -m experiments.runners.train_heads
python -m experiments.runners.train_uniform
```

After the reference states are available, the following command performs
RQ4 attention extraction and intervention evaluation without retraining:

```powershell
python -m experiments.analyse_fitted_gat
```

The repository already contains the completed results, so the training and
RQ4 analysis commands will refuse their existing output paths. Use the
saved-result commands above to reproduce the numerical summaries.

Git must be available for recorded fits and analysis because their source
commit is captured in the output. Commit the source and effective settings
and confirm a clean working tree before starting a recorded execution.

Each fit saves a paired JSON record and validation-selected .pt state.
Existing result or state paths are checked before fitting each model/dataset
group. Completed groups must be excluded explicitly when continuing a
partially completed matrix. Any deliberate rerun must preserve the original
evidence, use unused output paths and record its reason. Record results
after execution, retaining the source commit that produced the evidence.



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