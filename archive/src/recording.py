import json
import os
import subprocess

import torch
import torch_geometric


def get_source_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def get_result_path(settings, result_type):
    result_path = (
        f"results/{result_type}_"
        f"{settings['model'].lower()}_"
        f"{settings['dataset'].lower()}"
    )

    if settings.get("variant"):
        result_path += f"_{settings['variant'].lower()}"

    result_path += f"_seed{settings['seed']}.json"

    return result_path


def prepare_result_path(path):
    os.makedirs("results", exist_ok=True)

    if os.path.exists(path):
        raise FileExistsError(
            f"Result already exists: {path}"
        )


def save_result(path, result):
    result["pytorch_version"] = str(torch.__version__)
    result["pyg_version"] = torch_geometric.__version__
    result["cuda_version"] = torch.version.cuda

    with open(path, "x", encoding="utf-8") as result_file:
        json.dump(
            result,
            result_file,
            indent=4,
        )


def save_model_state(path, model):
    with open(path, "xb") as state_file:
        torch.save(
            model.state_dict(),
            state_file,
        )