from experiments.train import model_settings, settings
from src.recording import get_result_path


base_settings = settings.copy()

variants = [
    {
        "name": "GAT heads 1",
        "reuse_reference": False,
        "settings": {
            **base_settings,
            **model_settings["GAT"],
            "model": "GAT",
            "variant": "heads1",
            "heads": 1,
        },
    },
    {
        "name": "GAT heads 2",
        "reuse_reference": False,
        "settings": {
            **base_settings,
            **model_settings["GAT"],
            "model": "GAT",
            "variant": "heads2",
            "heads": 2,
        },
    },
    {
        "name": "GAT heads 4",
        "reuse_reference": False,
        "settings": {
            **base_settings,
            **model_settings["GAT"],
            "model": "GAT",
            "variant": "heads4",
            "heads": 4,
        },
    },
    {
        "name": "GAT heads 8 reference",
        "reuse_reference": True,
        "settings": {
            **base_settings,
            **model_settings["GAT"],
            "model": "GAT",
            "variant": None,
        },
    },
    {
        "name": "GATv2 reference",
        "reuse_reference": True,
        "settings": {
            **base_settings,
            **model_settings["GATv2"],
            "model": "GATv2",
            "variant": None,
        },
    },
    {
        "name": "Uniform GAT",
        "reuse_reference": False,
        "settings": {
            **base_settings,
            **model_settings["GAT"],
            "model": "GAT",
            "variant": "uniform",
            "uniform_attention": True,
        },
    },
]


def main():
    print("Core attention variants:")

    for variant in variants:
        current_settings = variant["settings"]

        heads = current_settings["heads"]
        hidden_dim = current_settings["hidden_dim"]
        channels_per_head = hidden_dim // heads

        print()
        print("Variant:", variant["name"])

        if variant["reuse_reference"]:
            print("Execution: reuse reference CV results")
        else:
            print("Execution: additional CV fit")

        print("Effective settings:")

        for name, value in current_settings.items():
            print(f"{name}: {value}")

        print("Derived channels per head:", channels_per_head)

        print(
            "Derived first-layer width:",
            channels_per_head * heads,
        )

        print("Final embedding width:", hidden_dim)

        print(
            "Current recorder path:",
            get_result_path(
                current_settings,
                "final",
            ),
        )


if __name__ == "__main__":
    main()