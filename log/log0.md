# 0.1 Repository

- Created the project folder

    - Named it Investigating Attention Mechanisms In Graph Neural Networks For Graph Classification - Version Infinity

    - The project investigates attention mechanisms in graph neural networks for graph classification

- Planned the initial repository files

    - README.md describes the project purpose and current development status

    - log/log0.md records repository setup, observations and decisions

- Selected a minimal .gitignore

    - .venv/ excludes the local Python virtual environment

    - data/ excludes downloaded datasets

    - __pycache__/ excludes generated Python caches

    - Additional ignore rules will be introduced when needed

- Introduced Git version control

    - Staging selects the file contents to include in the next commit

    - A commit preserves a snapshot of those contents with a descriptive message

    - The intended first commit message is 0.1: initialised project repository

- Execution

    - Git initialised and initial commit succeeded

    - Github repository created

    - git ls-files showed .gitignore, README.md and log/log0.md as the tracked files

    - git log -1 --oneline identified the initial commit as b0d1310

    - git status reported that main was up to date with origin/main and the working tree was clean





# 0.2 Environment and Device

- Inspected the existing Python installation and NVIDIA GPU

    - py --list identified Python 3.14, 64-bit

    - python --version reported Python 3.14.7

    - Test-Path .venv returned False before the virtual environment was created

    - nvidia-smi reported NVIDIA driver version 616.64 and CUDA UMD version 13.4

- Created and activated the project virtual environment

    - Used py -3.14 -m venv .venv to create the environment

    - Used .\.venv\Scripts\Activate.ps1 to activate it in PowerShell

    - The environment keeps the project's installed Python packages separate from other environments

    - The verification output confirmed that the running Python executable was inside the project's .venv\Scripts directory

- Installed the project packages

    - Used python -m pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130

    - Used python -m pip install numpy matplotlib "torch_geometric==2.8.0.post1"

    - The initial PyTorch installation attempt was cancelled before completion

    - Retried the installation successfully

- Recorded the actual environment versions

    - Python: 3.14.7

    - NumPy: 2.5.3

    - PyTorch: 2.13.0+cu130

    - PyTorch Geometric: 2.8.0.post1

    - Matplotlib: 3.11.1

    - PyTorch CUDA build: 13.0

    - The NVIDIA driver-side CUDA information and PyTorch's CUDA build are separate version values

- Verified package imports and a GPU calculation

    - The verification block imported platform, sys, Matplotlib, NumPy, PyTorch and PyTorch Geometric successfully

    - CUDA availability was True

    - PyTorch identified the GPU as NVIDIA GeForce RTX 4070 Laptop GPU

    - Created a two-by-two float32 tensor on the GPU

    - The tensor device was cuda:0

    - Multiplying the matrix [[1.0, 2.0], [3.0, 4.0]] by itself produced [[7.0, 10.0], [15.0, 22.0]]

    - This established that the environment could import the required packages and perform the small GPU calculation

- Explained the PowerShell syntax used for the verification block

    - @' opens a multiline string, and '@ closes it

    - The pipe symbol | passes the string to Python

    - python - reads and executes code from standard input, which receives the piped text

    - This allows a short Python script to run without saving a separate Python file

- Explained the environment information used in the verification block

    - platform is a standard-library module providing platform information; platform.python_version() returns the Python version

    - sys is a standard-library module providing interpreter information; sys.executable contains the path of the running Python executable

    - __version__ is an attribute provided by the installed packages that contains their version string

- Explained tensor conversion for displaying the result

    - .cpu() returns the tensor in CPU memory, copying it from GPU memory when necessary

    - .tolist() converts the tensor into ordinary Python values arranged in nested lists

    - In result.cpu().tolist(), the CPU transfer happens first, followed by conversion to lists

    - These operations return results without moving or replacing the original tensor in place





# 0.3 Package Pins and Minimal Layout

- Created requirements.txt using the versions verified in Stage 0.2

    - NumPy: 2.5.3

    - PyTorch: 2.13.0+cu130

    - PyTorch Geometric: 2.8.0.post1

    - Matplotlib: 3.11.1

    - The == notation pins a package to an exact version

    - The +cu130 suffix identifies the CUDA 13.0 PyTorch build

- Included the PyTorch package source in requirements.txt

    - Added --extra-index-url https://download.pytorch.org/whl/cu130

    - This makes the PyTorch CUDA package index available alongside the default Python Package Index, PyPI

    - All project packages can be installed with python -m pip install -r requirements.txt

    - The -r option tells pip to read requirements from the specified file

    - Pip also installs dependencies required by the listed packages

- Added environment setup instructions to README.md

    - Recorded the verified Python version, operating system, GPU and NVIDIA driver

    - Documented virtual environment creation and activation

    - Used one requirements-file installation command for the project packages

    - Explained that the virtual environment and downloaded datasets are excluded from Git

- Clarified Python installation and package installation

    - Python must be installed before creating a virtual environment or running pip

    - requirements.txt installs packages into an existing Python environment

    - It does not install the Python interpreter

    - Python 3.14 was selected to match the verified environment

- Defined the minimal package layout

    - experiments will contain executable project scripts

    - experiments/datasets will contain dataset inspection scripts

    - Empty __init__.py files will mark these directories as regular Python packages

    - Additional source files will be introduced when their functionality is needed

- Explained module execution

    - A module name uses dots to identify its position within packages

    - The future file experiments/datasets/inspect_mutag.py will have the module name experiments.datasets.inspect_mutag

    - python -m locates and executes a module

    - Project module commands will run from the repository root so Python can find the local packages

- Execution

    - Get-Content requirements.txt displayed the correct package pins and additional package index

    - Get-Content README.md displayed the environment setup instructions

    - Ran python -m pip install --dry-run -r requirements.txt

    - The --dry-run option reports what pip would install without installing packages

    - Pip recognised both PyPI and the PyTorch CUDA package index

    - All specified requirements were already satisfied in .venv

    - Package layout creation and the README development-status update are pending confirmation





# 0.4 Empty LaTeX Skeleton and Verified References

- Prepared the report source files

    - report/main.tex defines the report structure

    - report/references.bib stores bibliographic records

    - The repository will retain the current report source files

- Defined the empty report structure

    - Used the article document class with 11-point text and A4 paper

    - Included Abstract, Introduction, Background and Related Work, Methodology, Results, Discussion, Limitations and Conclusion

    - Substantive report writing remains scheduled for the dedicated report phase

- Selected the LaTeX packages

    - amsmath provides mathematical typesetting tools

    - graphicx supports figures

    - booktabs supports table rules

    - natbib provides citations, with numerical citation formatting selected

- Prepared two verified bibliography entries

    - Semi-Supervised Classification with Graph Convolutional Networks, by Thomas N. Kipf and Max Welling, ICLR 2017

    - Graph Attention Networks, by Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò and Yoshua Bengio, ICLR 2018

    - Titles, author order and publication years were checked against the supplied papers and primary records

- Explained bibliography handling

    - Citation keys identify records in references.bib

    - citep inserts a citation into manuscript text

    - A temporary nocite command includes the two references in the empty skeleton

    - The unsrtnat bibliography style formats references in citation order

- Selected Overleaf for compilation

    - Compilation converts the LaTeX source into a PDF

    - Changes made in Overleaf must also be saved in the repository copies

- Execution

    - File creation, compilation and inspection of the resulting PDF are pending confirmation