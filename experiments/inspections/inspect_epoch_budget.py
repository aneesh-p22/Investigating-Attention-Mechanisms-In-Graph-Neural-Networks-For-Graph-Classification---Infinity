import time

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from src.data import load_dataset, set_seed, stratified_split
from src.evaluation import evaluate
from src.models.factory import build_model
from src.recording import (
    get_source_commit,
    prepare_result_path,
    save_result,
)
from src.training import train_epoch


datasets = [
    "MUTAG",
    "PROTEINS",
    "NCI1",
]

configurations = [
    {
        "name": "GCN",
        "settings": {
            "model": "GCN",
            "variant": None,
        },
    },
    {
        "name": "GraphSAGE",
        "settings": {
            "model": "GraphSAGE",
            "variant": None,
        },
    },
    {
        "name": "GIN",
        "settings": {
            "model": "GIN",
            "variant": None,
        },
    },
    {
        "name": "GAT heads 1",
        "settings": {
            "model": "GAT",
            "variant": "heads1",
            "heads": 1,
        },
    },
    {
        "name": "GAT heads 2",
        "settings": {
            "model": "GAT",
            "variant": "heads2",
            "heads": 2,
        },
    },
    {
        "name": "GAT heads 4",
        "settings": {
            "model": "GAT",
            "variant": "heads4",
            "heads": 4,
        },
    },
    {
        "name": "GAT heads 8",
        "settings": {
            "model": "GAT",
            "variant": None,
            "heads": 8,
        },
    },
    {
        "name": "GATv2",
        "settings": {
            "model": "GATv2",
            "variant": None,
            "heads": 8,
            "share_weights": False,
        },
    },
    {
        "name": "Uniform GAT",
        "settings": {
            "model": "GAT",
            "variant": "uniform",
            "heads": 8,
            "uniform_attention": True,
        },
    },
]

settings = {
    "hidden_dim": 64,
    "learning_rate": 0.01,
    "weight_decay": 0.0005,
    "epochs": 1000,
    "batch_size": 32,
    "seed": 0,
    "split_seed": 0,
}

cutoffs = [
    100,
    250,
    500,
    1000,
]

top_epoch_count = 5

result_path = "results/development_epoch_budget_audit.json"


def validation_sort_key(record):
    return (
        record["validation_loss"],
        record["epoch"],
    )


def run_configuration(
    dataset,
    configuration,
    device,
):
    current_settings = settings.copy()
    current_settings.update(configuration["settings"])
    current_settings["dataset"] = dataset.name

    train_indices, val_indices, _ = stratified_split(
        dataset,
        current_settings["split_seed"],
    )

    set_seed(current_settings["seed"])

    generator = torch.Generator()
    generator.manual_seed(current_settings["seed"])

    train_loader = DataLoader(
        dataset[train_indices],
        batch_size=current_settings["batch_size"],
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        dataset[val_indices],
        batch_size=current_settings["batch_size"],
        shuffle=False,
    )

    model = build_model(
        current_settings,
        dataset.num_node_features,
        dataset.num_classes,
    ).to(device)

    optimizer = Adam(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=current_settings["learning_rate"],
        weight_decay=current_settings["weight_decay"],
    )

    best_epoch = 0
    best_val_loss = float("inf")
    best_val_accuracy = 0.0

    validation_history = []
    cutoff_records = {}

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for epoch in range(1, current_settings["epochs"] + 1):
        train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_loss, val_accuracy, _, _ = evaluate(
            model,
            val_loader,
            device,
        )

        validation_history.append(
            {
                "epoch": epoch,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
            }
        )

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy

        if epoch in cutoffs:
            if device.type == "cuda":
                torch.cuda.synchronize()

            cutoff_records[epoch] = {
                "selected_epoch": best_epoch,
                "validation_loss": best_val_loss,
                "validation_accuracy": best_val_accuracy,
                "elapsed_seconds": (
                    time.perf_counter() - start_time
                ),
            }

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_seconds = time.perf_counter() - start_time

    ranked_epochs = sorted(
        validation_history,
        key=validation_sort_key,
    )

    top_epochs = ranked_epochs[:top_epoch_count]

    for cutoff in cutoffs:
        cutoff_records[cutoff]["loss_gap"] = (
            cutoff_records[cutoff]["validation_loss"]
            - best_val_loss
        )

    return {
        "name": configuration["name"],
        "settings": current_settings,
        "training_graphs": len(train_indices),
        "validation_graphs": len(val_indices),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": best_val_accuracy,
        "top_epochs": top_epochs,
        "cutoffs": {
            str(cutoff): cutoff_records[cutoff]
            for cutoff in cutoffs
        },
        "validation_history": validation_history,
        "elapsed_seconds": elapsed_seconds,
    }


def print_run(result):
    print()
    print("=" * 70)
    print("Dataset:", result["settings"]["dataset"])
    print("Configuration:", result["name"])
    print("Training graphs:", result["training_graphs"])
    print("Validation graphs:", result["validation_graphs"])

    print()
    print("Overall best validation state:")
    print("Selected epoch:", result["best_epoch"])

    print(
        f"Validation loss: "
        f"{result['best_validation_loss']:.6f}"
    )

    print(
        f"Validation accuracy: "
        f"{result['best_validation_accuracy']:.4f}"
    )

    print()
    print(f"Top {top_epoch_count} validation epochs:")

    for rank, record in enumerate(
        result["top_epochs"],
        start=1,
    ):
        print(
            f"{rank}. "
            f"epoch {record['epoch']} | "
            f"loss {record['validation_loss']:.6f} | "
            f"accuracy {record['validation_accuracy']:.4f}"
        )

    print()
    print("Best validation state available by cutoff:")

    for cutoff in cutoffs:
        record = result["cutoffs"][str(cutoff)]

        print(
            f"{cutoff:4d} epochs | "
            f"selected {record['selected_epoch']:4d} | "
            f"loss {record['validation_loss']:.6f} | "
            f"gap {record['loss_gap']:+.6f} | "
            f"accuracy {record['validation_accuracy']:.4f} | "
            f"elapsed {record['elapsed_seconds']:.2f}s"
        )

    print()
    print(
        "Full elapsed seconds:",
        f"{result['elapsed_seconds']:.2f}",
    )


def main():
    prepare_result_path(result_path)

    source_commit = get_source_commit()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Source commit:", source_commit)
    print("Device:", device)

    results = []

    audit_start = time.perf_counter()

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)

        for configuration in configurations:
            result = run_configuration(
                dataset,
                configuration,
                device,
            )

            results.append(result)
            print_run(result)

    if device.type == "cuda":
        torch.cuda.synchronize()

    audit_seconds = time.perf_counter() - audit_start

    result = {
        "purpose": "development epoch budget audit",
        "source_commit": source_commit,
        "datasets": datasets,
        "configurations": [
            configuration["name"]
            for configuration in configurations
        ],
        "settings": settings,
        "cutoffs": cutoffs,
        "top_epoch_count": top_epoch_count,
        "runs": results,
        "audit_seconds": audit_seconds,
        "interpretation": (
            "Development validation evidence only. "
            "No development-test or final-test results are used "
            "to choose the common final epoch budget."
        ),
    }

    save_result(
        result_path,
        result,
    )

    print()
    print("=" * 70)
    print("Audit complete")
    print("Runs:", len(results))

    print(
        "Total elapsed seconds:",
        f"{audit_seconds:.2f}",
    )

    print(
        "Total elapsed hours:",
        f"{audit_seconds / 3600:.2f}",
    )

    print("Saved result:", result_path)


if __name__ == "__main__":
    main()