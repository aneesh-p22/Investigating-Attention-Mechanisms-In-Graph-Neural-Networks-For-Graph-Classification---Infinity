import json
import math
import os

import numpy as np

from experiments.train_cv import (
    datasets,
    model_settings,
    models,
    result_type,
    settings,
)
from src.data import (
    load_dataset,
    stratified_folds,
    stratified_validation_split,
)
from src.recording import get_result_path


def get_partitions(dataset):
    outer_folds = stratified_folds(
        dataset,
        settings["num_folds"],
        settings["split_seed"],
    )

    partitions = []

    for fold_id, test_indices in enumerate(outer_folds):
        test_set = set(test_indices)
        remainder_indices = [
            index
            for index in range(len(dataset))
            if index not in test_set
        ]

        fit_indices, val_indices = stratified_validation_split(
            dataset,
            remainder_indices,
            1000 + fold_id,
        )

        partitions.append(
            {
                "outer_split_seed": settings["split_seed"],
                "validation_split_seed": 1000 + fold_id,
                "fit_indices": fit_indices,
                "validation_indices": val_indices,
                "test_indices": test_indices,
            }
        )

    return partitions


def load_group(current_settings, dataset, partitions):
    prefix = (
        f"{result_type}_"
        f"{current_settings['model'].lower()}_"
        f"{current_settings['dataset'].lower()}_fold"
    )

    results = {}

    for filename in sorted(os.listdir("results")):
        if not filename.startswith(prefix) or not filename.endswith(".json"):
            continue

        result_path = os.path.join("results", filename)

        with open(result_path, encoding="utf-8") as result_file:
            result = json.load(result_file)

        fold_id = result["settings"]["fold_id"]

        if type(fold_id) is not int or fold_id not in range(settings["num_folds"]):
            raise ValueError(f"Unexpected outer fold: {result_path}")

        if fold_id in results:
            raise ValueError(f"Duplicate outer fold: {result_path}")

        expected_settings = current_settings.copy()
        expected_settings["fold_id"] = fold_id
        expected_settings["validation_split_seed"] = 1000 + fold_id
        expected_settings["seed"] = 2000 + fold_id

        if result["purpose"] != result_type or result["settings"] != expected_settings:
            raise ValueError(f"Inconsistent purpose or settings: {result_path}")

        expected_path = get_result_path(expected_settings, result_type)

        if os.path.normpath(result_path) != os.path.normpath(expected_path):
            raise ValueError(f"Filename does not match settings: {result_path}")

        state_path = expected_path.replace(".json", ".pt")

        if os.path.normpath(result["model_state_path"]) != os.path.normpath(state_path):
            raise ValueError(f"Incorrect paired state path: {result_path}")

        if not os.path.isfile(state_path):
            raise FileNotFoundError(state_path)

        if result["dataset"] != {
            "cleaned": False,
            "use_node_attr": False,
            "use_edge_attr": False,
        } or result["feature_policy"] != {
            "node_labels_used": True,
            "node_attributes_used": False,
            "connectivity_used": True,
            "edge_features_used": False,
        }:
            raise ValueError(f"Inconsistent dataset or feature policy: {result_path}")

        if result["partitions"] != partitions[fold_id]:
            raise ValueError(f"Incorrect partition indices or seeds: {result_path}")

        selection = result["selection"]

        if (
            selection["criterion"] != "minimum validation cross-entropy"
            or selection["tie_rule"] != "earliest exact tie"
            or selection["early_stopping"] is not False
            or selection["completed_epochs"] != current_settings["epochs"]
            or not 1 <= selection["selected_epoch"] <= selection["completed_epochs"]
        ):
            raise ValueError(f"Inconsistent checkpoint selection: {result_path}")

        outer_test = result["outer_test"]
        predictions = outer_test["predictions"]
        labels = outer_test["labels"]
        expected_labels = [
            dataset[index].y.item()
            for index in partitions[fold_id]["test_indices"]
        ]

        if labels != expected_labels or len(predictions) != len(labels):
            raise ValueError(f"Misaligned test predictions or labels: {result_path}")

        if any(
            type(prediction) is not int or not 0 <= prediction < dataset.num_classes
            for prediction in predictions
        ):
            raise ValueError(f"Invalid predicted class: {result_path}")

        accuracy = sum(
            prediction == label
            for prediction, label in zip(predictions, labels)
        ) / len(labels)

        if not math.isclose(accuracy, outer_test["accuracy"], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Stored accuracy disagrees with predictions: {result_path}")

        for loss in [selection["validation_loss"], outer_test["loss"]]:
            if not math.isfinite(loss) or loss < 0:
                raise ValueError(f"Invalid recorded loss: {result_path}")

        if not 0 <= selection["validation_accuracy"] <= 1:
            raise ValueError(f"Invalid validation accuracy: {result_path}")

        parameters = result["parameters"]

        if not 0 < parameters["trainable"] <= parameters["total"]:
            raise ValueError(f"Invalid parameter counts: {result_path}")

        runtime = result["runtime"]
        training_seconds = runtime["training_seconds"]

        if not math.isfinite(training_seconds) or training_seconds < 0:
            raise ValueError(f"Invalid training time: {result_path}")

        if not math.isclose(
            runtime["mean_seconds_per_epoch"],
            training_seconds / selection["completed_epochs"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Inconsistent mean epoch time: {result_path}")

        results[fold_id] = result

    if set(results) != set(range(settings["num_folds"])):
        raise ValueError(f"Incomplete outer-fold group: {prefix}")

    return [results[fold_id] for fold_id in range(settings["num_folds"])]


def main():
    groups = []

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        partitions = get_partitions(dataset)

        for model_name in models:
            current_settings = settings.copy()
            current_settings.update(model_settings[model_name])
            current_settings["dataset"] = dataset_name
            current_settings["model"] = model_name

            results = load_group(
                current_settings,
                dataset,
                partitions,
            )

            if any(result["parameters"] != results[0]["parameters"] for result in results):
                raise ValueError(f"Parameter counts differ: {dataset_name}, {model_name}")

            groups.append(results)

    all_results = [result for group in groups for result in group]
    first_result = all_results[0]

    for result in all_results:
        for name in ["device", "pytorch_version", "pyg_version", "cuda_version"]:
            if result[name] != first_result[name]:
                raise ValueError(f"Inconsistent runtime conditions: {name}")

        if result["runtime"]["convention"] != first_result["runtime"]["convention"]:
            raise ValueError("Inconsistent runtime conventions")

    print("Validated reference fits:", len(all_results))
    print("Dataset/model groups:", len(groups))
    print("Outer folds per group:", settings["num_folds"])
    print("Device:", first_result["device"])
    print("PyTorch:", first_result["pytorch_version"])
    print("PyG:", first_result["pyg_version"])
    print("CUDA build:", first_result["cuda_version"])
    print("Source commits:")

    for source_commit in sorted({result["source_commit"] for result in all_results}):
        print(source_commit)

    print()
    print("Outer-test accuracy (%):")
    fold_columns = " | ".join(
        f"Outer fold {fold_id}" for fold_id in range(settings["num_folds"])
    )
    print(f"Dataset | Model | {fold_columns} | Mean | Sample SD")

    for results in groups:
        current_settings = results[0]["settings"]
        accuracies = np.array(
            [result["outer_test"]["accuracy"] for result in results]
        ) * 100
        fold_values = " | ".join(f"{accuracy:.2f}" for accuracy in accuracies)

        print(
            f"{current_settings['dataset']} | {current_settings['model']} | "
            f"{fold_values} | {accuracies.mean():.2f} | {accuracies.std(ddof=1):.2f}"
        )

    print()
    print("Parameters and training time:")
    print("Training passes only; time totals cover all outer folds.")
    print(
        "Dataset | Model | Total parameters | Trainable parameters | "
        "Total time (s) | Mean epoch time (s)"
    )

    for results in groups:
        current_settings = results[0]["settings"]
        parameters = results[0]["parameters"]
        training_seconds = sum(result["runtime"]["training_seconds"] for result in results)
        completed_epochs = sum(result["selection"]["completed_epochs"] for result in results)

        print(
            f"{current_settings['dataset']} | {current_settings['model']} | "
            f"{parameters['total']} | {parameters['trainable']} | "
            f"{training_seconds:.2f} | {training_seconds / completed_epochs:.4f}"
        )


if __name__ == "__main__":
    main()