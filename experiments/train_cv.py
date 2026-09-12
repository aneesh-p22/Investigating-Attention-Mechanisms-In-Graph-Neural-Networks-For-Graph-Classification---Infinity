import math

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from src.data import (
    load_dataset,
    set_seed,
    stratified_folds,
    stratified_validation_split,
)
from src.evaluation import evaluate
from src.models.factory import build_model
from src.recording import (
    get_result_path,
    get_source_commit,
    prepare_result_path,
    save_model_state,
    save_result,
)
from src.training import train_model


settings = {
    "variant": None,
    "hidden_dim": 64,
    "learning_rate": 0.01,
    "weight_decay": 0.0005,
    "epochs": 500,
    "batch_size": 32,
    "split_seed": 0,
    "num_folds": 5,
}

model_settings = {
    "GCN": {},
    "GraphSAGE": {},
    "GIN": {},
    "GAT": {
        "heads": 8,
    },
    "GATv2": {
        "heads": 8,
        "share_weights": False,
    },
}

datasets = [
    "MUTAG",
    "PROTEINS",
    "NCI1",
]

models = [
    "GCN",
    "GraphSAGE",
    "GIN",
    "GAT",
    "GATv2",
]

fold_ids = list(range(settings["num_folds"]))

result_type = "cross_validation"


def run_cross_validation(current_settings, fold_ids):
    dataset = load_dataset(current_settings["dataset"])

    outer_folds = stratified_folds(
        dataset,
        current_settings["num_folds"],
        current_settings["split_seed"],
    )

    fold_runs = []

    for fold_id in fold_ids:
        fold_settings = current_settings.copy()
        fold_settings["fold_id"] = fold_id
        fold_settings["validation_split_seed"] = 1000 + fold_id
        fold_settings["seed"] = 2000 + fold_id

        result_path = get_result_path(
            fold_settings,
            result_type,
        )

        state_path = result_path.replace(".json", ".pt")

        prepare_result_path(result_path)
        prepare_result_path(state_path)

        fold_runs.append(
            {
                "settings": fold_settings,
                "result_path": result_path,
                "state_path": state_path,
            }
        )

    source_commit = get_source_commit()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("Source commit:", source_commit)
    print("Device:", device)
    print("Dataset:", current_settings["dataset"])
    print("Model:", current_settings["model"])

    for fold_run in fold_runs:
        fold_settings = fold_run["settings"]
        result_path = fold_run["result_path"]
        state_path = fold_run["state_path"]

        fold_id = fold_settings["fold_id"]
        test_indices = outer_folds[fold_id]
        test_set = set(test_indices)

        remainder_indices = [
            index
            for index in range(len(dataset))
            if index not in test_set
        ]

        fit_indices, val_indices = stratified_validation_split(
            dataset,
            remainder_indices,
            fold_settings["validation_split_seed"],
        )

        set_seed(fold_settings["seed"])

        generator = torch.Generator()
        generator.manual_seed(fold_settings["seed"])

        fit_loader = DataLoader(
            dataset[fit_indices],
            batch_size=fold_settings["batch_size"],
            shuffle=True,
            generator=generator,
        )

        val_loader = DataLoader(
            dataset[val_indices],
            batch_size=fold_settings["batch_size"],
            shuffle=False,
        )

        test_loader = DataLoader(
            dataset[test_indices],
            batch_size=fold_settings["batch_size"],
            shuffle=False,
        )

        model = build_model(
            fold_settings,
            dataset.num_node_features,
            dataset.num_classes,
        ).to(device)

        optimizer = Adam(
            (
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            lr=fold_settings["learning_rate"],
            weight_decay=fold_settings["weight_decay"],
        )

        print()
        print("Outer fold:", fold_id)
        print("Fit graphs:", len(fit_indices))
        print("Validation graphs:", len(val_indices))
        print("Outer test graphs:", len(test_indices))
        print("Training seed:", fold_settings["seed"])
        print(
            "Validation split seed:",
            fold_settings["validation_split_seed"],
        )

        print()
        print("Training:")

        (
            best_epoch,
            best_val_loss,
            best_val_accuracy,
            training_seconds,
        ) = train_model(
            model,
            fit_loader,
            val_loader,
            optimizer,
            device,
            fold_settings["epochs"],
        )

        (
            test_loss,
            test_accuracy,
            test_predictions,
            test_labels,
        ) = evaluate(
            model,
            test_loader,
            device,
        )

        if not math.isfinite(test_loss):
            raise ValueError(f"Non-finite outer-test loss in outer fold {fold_id}")

        total_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        mean_seconds_per_epoch = (
            training_seconds / fold_settings["epochs"]
        )

        if device.type == "cuda":
            device_name = torch.cuda.get_device_name(device)
        else:
            device_name = "CPU"

        result = {
            "purpose": result_type,
            "settings": fold_settings,
            "dataset": {
                "cleaned": False,
                "use_node_attr": False,
                "use_edge_attr": False,
            },
            "feature_policy": {
                "node_labels_used": True,
                "node_attributes_used": False,
                "connectivity_used": True,
                "edge_features_used": False,
            },
            "partitions": {
                "outer_split_seed": fold_settings["split_seed"],
                "validation_split_seed": (
                    fold_settings["validation_split_seed"]
                ),
                "fit_indices": fit_indices,
                "validation_indices": val_indices,
                "test_indices": test_indices,
            },
            "selection": {
                "criterion": "minimum validation cross-entropy",
                "tie_rule": "earliest exact tie",
                "early_stopping": False,
                "selected_epoch": best_epoch,
                "completed_epochs": fold_settings["epochs"],
                "validation_loss": best_val_loss,
                "validation_accuracy": best_val_accuracy,
            },
            "outer_test": {
                "loss": test_loss,
                "accuracy": test_accuracy,
                "predictions": test_predictions,
                "labels": test_labels,
            },
            "parameters": {
                "total": total_parameters,
                "trainable": trainable_parameters,
            },
            "runtime": {
                "training_seconds": training_seconds,
                "mean_seconds_per_epoch": mean_seconds_per_epoch,
                "convention": (
                    "train_epoch call only; includes loader iteration, "
                    "device transfer, forward pass, loss, backward pass, "
                    "optimiser update and metric calculation and "
                    "accumulation; excludes validation, outer test, "
                    "checkpoint copying and restoration, progress "
                    "printing, model-state saving and result writing"
                ),
            },
            "device": {
                "type": str(device),
                "name": device_name,
            },
            "source_commit": source_commit,
            "model_state_path": state_path,
        }

        save_model_state(
            state_path,
            model,
        )

        save_result(
            result_path,
            result,
        )

        print()
        print("Selected state:")
        print("Selected epoch:", best_epoch)
        print(f"Validation loss: {best_val_loss:.4f}")
        print(f"Validation accuracy: {best_val_accuracy:.4f}")

        print()
        print("Outer test:")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")

        print()
        print("Total parameters:", total_parameters)
        print("Trainable parameters:", trainable_parameters)
        print(f"Training seconds: {training_seconds:.4f}")
        print(
            f"Mean seconds per epoch: "
            f"{mean_seconds_per_epoch:.4f}"
        )

        print()
        print("Saved model state:", state_path)
        print("Saved result:", result_path)


def main():
    for dataset_name in datasets:
        for model_name in models:
            current_settings = settings.copy()
            current_settings.update(model_settings[model_name])
            current_settings["dataset"] = dataset_name
            current_settings["model"] = model_name

            run_cross_validation(
                current_settings,
                fold_ids,
            )


if __name__ == "__main__":
    main()