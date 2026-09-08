import json
import subprocess

import torch
import torch_geometric


def get_source_commit():
    """Return the current Git commit after requiring committed source."""
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
    ).strip()

    if status:
        raise RuntimeError(
            "Commit the source and settings before running a recorded fit."
        )

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


def check_result_path(path):
    """Refuse to reuse an existing result filename."""
    try:
        with open(path, "r", encoding="utf-8"):
            pass
    except FileNotFoundError:
        return

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