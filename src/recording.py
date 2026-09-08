import json
import os
import subprocess

import torch
import torch_geometric


def get_source_commit():
    """Return the current Git commit."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def get_result_path(settings, result_type):
    """Return the filename for one recorded result."""
    filename = (
        f"results/{result_type}_"
        f"{settings['model'].lower()}_"
        f"{settings['dataset'].lower()}"
    )

    if settings.get("variant"):
        filename += f"_{settings['variant'].lower()}"

    filename += f"_seed{settings['seed']}.json"

    return filename


def prepare_result_path(path):
    """Prepare the results directory and refuse an existing filename."""
    os.makedirs("results", exist_ok=True)

    if os.path.exists(path):
        raise FileExistsError(
            f"Result already exists: {path}"
        )


def save_result(path, result):
    """Save one result without overwriting an existing file."""
    result["pytorch_version"] = str(torch.__version__)
    result["pyg_version"] = torch_geometric.__version__
    result["cuda_version"] = torch.version.cuda

    with open(path, "x", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=4)