# Investigating Attention Mechanisms in Graph Neural Networks for Graph Classification

EPSRC summer research internship at the University of Leicester.

Researcher: Aneesh  
Supervisor: Dr Furqan Aziz



## Purpose

Investigate graph attention design using GAT and GATv2, with GCN,
GraphSAGE and GIN as contextual baselines, on MUTAG, PROTEINS and NCI1.



## Development status

Stage 0: repository and environment setup.

Implementation, environment verification and experiments for this fresh
project are pending.



## Project guidance

- log/ records development observations and decisions.



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