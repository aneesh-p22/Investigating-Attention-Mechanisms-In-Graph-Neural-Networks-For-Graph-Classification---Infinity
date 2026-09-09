import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from src.data import load_dataset, set_seed, stratified_split
from src.models.factory import build_model
from src.evaluation import evaluate
from src.recording import (
    get_result_path,
    get_source_commit,
    prepare_result_path,
    save_result,
)
from src.training import train_model


settings = {
    "variant": None,
    "dataset": "MUTAG",
    "baseline_width": 64,
    "learning_rate": 0.01,
    "weight_decay": 0.0005,
    "epochs": 1000,
    "batch_size": 32,
    "seed": 0,
    "split_seed": 0,
}

model_settings = {
    "GCN": {},
    "GraphSAGE": {},
    "GIN": {},
}

selected_models = [
    "GCN",
    "GraphSAGE",
    "GIN",
]


def main():
    result_type = "development"
    run_settings = []

    for model_name in selected_models:
        current_settings = settings.copy()
        current_settings.update(model_settings[model_name])
        current_settings["model"] = model_name

        result_path = get_result_path(
            current_settings,
            result_type,
        )

        prepare_result_path(result_path)
        run_settings.append(current_settings)

    source_commit = get_source_commit()

    dataset = load_dataset(settings["dataset"])

    train_indices, val_indices, test_indices = stratified_split(
        dataset,
        settings["split_seed"],
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Source commit: {source_commit}")
    print(f"Device: {device}")
    print(f"Training graphs: {len(train_indices)}")
    print(f"Validation graphs: {len(val_indices)}")
    print(f"Development test graphs: {len(test_indices)}")

    results = []

    for current_settings in run_settings:
        if (
            current_settings["dataset"] != settings["dataset"]
            or current_settings["split_seed"] != settings["split_seed"]
        ):
            raise ValueError(
                "Dataset and split seed must remain shared across models."
            )

        result_path = get_result_path(
            current_settings,
            result_type,
        )

        set_seed(current_settings["seed"])

        train_generator = torch.Generator()
        train_generator.manual_seed(current_settings["seed"])

        train_loader = DataLoader(
            dataset[train_indices],
            batch_size=current_settings["batch_size"],
            shuffle=True,
            generator=train_generator,
        )

        val_loader = DataLoader(
            dataset[val_indices],
            batch_size=current_settings["batch_size"],
            shuffle=False,
        )

        test_loader = DataLoader(
            dataset[test_indices],
            batch_size=current_settings["batch_size"],
            shuffle=False,
        )

        model = build_model(
            current_settings,
            dataset.num_node_features,
            dataset.num_classes,
        ).to(device)

        optimizer = Adam(
            model.parameters(),
            lr=current_settings["learning_rate"],
            weight_decay=current_settings["weight_decay"],
        )

        print()
        print("Settings:")

        for name, value in current_settings.items():
            print(f"{name}: {value}")

        print()
        print("Training:")

        (
            best_epoch,
            best_val_loss,
            best_val_accuracy,
            training_seconds,
        ) = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            current_settings["epochs"],
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            device,
        )

        total_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        mean_training_seconds = (
            training_seconds / current_settings["epochs"]
        )

        if device.type == "cuda":
            device_name = torch.cuda.get_device_name(device)
        else:
            device_name = "CPU"

        result = {
            "purpose": result_type,
            "settings": current_settings,
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
                "train_indices": train_indices,
                "validation_indices": val_indices,
                "test_indices": test_indices,
            },
            "selection": {
                "criterion": "minimum validation cross-entropy",
                "tie_rule": "earliest exact tie",
                "early_stopping": False,
                "selected_epoch": best_epoch,
                "completed_epochs": current_settings["epochs"],
                "validation_loss": best_val_loss,
                "validation_accuracy": best_val_accuracy,
            },
            "development_test": {
                "loss": test_loss,
                "accuracy": test_accuracy,
            },
            "parameters": {
                "total": total_parameters,
                "trainable": trainable_parameters,
            },
            "runtime": {
                "training_seconds": training_seconds,
                "mean_seconds_per_epoch": mean_training_seconds,
                "convention": (
                    "training pass only; includes loader iteration, "
                    "device transfer, forward pass, loss, backward pass "
                    "and optimiser update; excludes validation, "
                    "development test and result writing"
                ),
            },
            "device": {
                "type": str(device),
                "name": device_name,
            },
            "source_commit": source_commit,
        }

        print()
        print("Selected state:")
        print(f"Selected epoch: {best_epoch}")
        print(f"Selected validation loss: {best_val_loss:.4f}")
        print(f"Selected validation accuracy: {best_val_accuracy:.4f}")

        print()
        print("Development test:")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")

        print()
        print(f"Total parameters: {total_parameters}")
        print(f"Trainable parameters: {trainable_parameters}")
        print(f"Training seconds: {training_seconds:.4f}")
        print(
            f"Mean training seconds per epoch: "
            f"{mean_training_seconds:.4f}"
        )

        save_result(
            result_path,
            result,
        )

        print(f"Saved result: {result_path}")

        results.append(result)

    print()
    print("Completed development fits:")

    for result in results:
        selection = result["selection"]
        parameters = result["parameters"]
        development_test = result["development_test"]
        runtime = result["runtime"]

        print(
            f"{result['settings']['model']}: "
            f"selected epoch {selection['selected_epoch']}, "
            f"val loss {selection['validation_loss']:.4f}, "
            f"val accuracy {selection['validation_accuracy']:.4f}, "
            f"development test accuracy {development_test['accuracy']:.4f}, "
            f"parameters {parameters['total']}, "
            f"seconds per epoch {runtime['mean_seconds_per_epoch']:.4f}"
        )


if __name__ == "__main__":
    main()