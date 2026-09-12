from experiments.train_cv import (
    datasets,
    fold_ids,
    model_settings,
    result_type,
    run_cross_validation,
    settings,
)
from src.recording import get_result_path


variants = [
    {
        "name": "GAT heads 1",
        "reuse_reference": False,
        "settings": {
            "model": "GAT",
            "variant": "heads1",
            "heads": 1,
        },
    },
    {
        "name": "GAT heads 2",
        "reuse_reference": False,
        "settings": {
            "model": "GAT",
            "variant": "heads2",
            "heads": 2,
        },
    },
    {
        "name": "GAT heads 4",
        "reuse_reference": False,
        "settings": {
            "model": "GAT",
            "variant": "heads4",
            "heads": 4,
        },
    },
    {
        "name": "GAT heads 8 reference",
        "reuse_reference": True,
        "settings": {
            "model": "GAT",
            "variant": None,
        },
    },
    {
        "name": "GATv2 reference",
        "reuse_reference": True,
        "settings": {
            "model": "GATv2",
            "variant": None,
        },
    },
    {
        "name": "Uniform GAT",
        "reuse_reference": False,
        "settings": {
            "model": "GAT",
            "variant": "uniform",
            "uniform_attention": True,
        },
    },
]


def get_variant_settings(variant):
    current_settings = settings.copy()

    model_name = variant["settings"]["model"]
    current_settings.update(model_settings[model_name])
    current_settings.update(variant["settings"])

    return current_settings


def run_variant(variant):
    if variant["reuse_reference"]:
        raise ValueError("Reference variants reuse existing cross-validation results")

    for dataset_name in datasets:
        current_settings = get_variant_settings(variant)
        current_settings["dataset"] = dataset_name

        run_cross_validation(
            current_settings,
            fold_ids,
        )


def main():
    print("Core attention variants:")

    for variant in variants:
        current_settings = get_variant_settings(variant)

        heads = current_settings["heads"]
        hidden_dim = current_settings["hidden_dim"]
        channels_per_head = hidden_dim // heads

        print()
        print("Variant:", variant["name"])

        if variant["reuse_reference"]:
            print("Execution: reuse cross-validation results")
        else:
            print("Execution: additional cross-validation fit")

        print("Effective settings:")

        for name, value in current_settings.items():
            print(f"{name}: {value}")

        print("Derived channels per head:", channels_per_head)
        print(
            "Derived first-layer width:",
            channels_per_head * heads,
        )
        print("Embedding width:", hidden_dim)

        fold_settings = current_settings.copy()
        fold_settings["dataset"] = datasets[0]
        fold_settings["fold_id"] = fold_ids[0]
        fold_settings["seed"] = 2000 + fold_ids[0]

        print(
            "Example result path:",
            get_result_path(
                fold_settings,
                result_type,
            ),
        )


if __name__ == "__main__":
    main()