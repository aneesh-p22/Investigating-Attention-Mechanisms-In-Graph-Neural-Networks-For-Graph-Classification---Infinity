import json
import math
import os

from experiments.train_cv import result_type, settings
from src.data import (
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

        fit_indices, validation_indices = stratified_validation_split(
            dataset,
            remainder_indices,
            1000 + fold_id,
        )

        partitions.append(
            {
                "outer_split_seed": settings["split_seed"],
                "validation_split_seed": 1000 + fold_id,
                "fit_indices": fit_indices,
                "validation_indices": validation_indices,
                "test_indices": test_indices,
            }
        )

    return partitions


def load_group(current_settings, dataset, partitions):
    results = []

    for fold_id in range(settings["num_folds"]):
        expected_settings = current_settings.copy()
        expected_settings["fold_id"] = fold_id
        expected_settings["validation_split_seed"] = 1000 + fold_id
        expected_settings["seed"] = 2000 + fold_id

        result_path = get_result_path(
            expected_settings,
            result_type,
        )

        with open(result_path, encoding="utf-8") as result_file:
            result = json.load(result_file)

        if result["purpose"] != result_type:
            raise ValueError(
                f"Incorrect result purpose: {result_path}"
            )

        if result["settings"] != expected_settings:
            raise ValueError(
                f"Incorrect result settings: {result_path}"
            )

        state_path = result_path.replace(".json", ".pt")

        if os.path.normpath(
            result["model_state_path"]
        ) != os.path.normpath(state_path):
            raise ValueError(
                f"Incorrect paired state path: {result_path}"
            )

        if not os.path.isfile(state_path):
            raise FileNotFoundError(state_path)

        if result["dataset"] != {
            "cleaned": False,
            "use_node_attr": False,
            "use_edge_attr": False,
        }:
            raise ValueError(
                f"Incorrect dataset settings: {result_path}"
            )

        if result["feature_policy"] != {
            "node_labels_used": True,
            "node_attributes_used": False,
            "connectivity_used": True,
            "edge_features_used": False,
        }:
            raise ValueError(
                f"Incorrect feature policy: {result_path}"
            )

        if result["partitions"] != partitions[fold_id]:
            raise ValueError(
                f"Incorrect partition indices or seeds: {result_path}"
            )

        selection = result["selection"]

        if (
            selection["criterion"]
            != "minimum validation cross-entropy"
            or selection["tie_rule"] != "earliest exact tie"
            or selection["early_stopping"] is not False
            or selection["completed_epochs"]
            != current_settings["epochs"]
            or not 1
            <= selection["selected_epoch"]
            <= selection["completed_epochs"]
        ):
            raise ValueError(
                f"Incorrect checkpoint selection: {result_path}"
            )

        outer_test = result["outer_test"]
        predictions = outer_test["predictions"]
        labels = outer_test["labels"]

        expected_labels = [
            dataset[index].y.item()
            for index in partitions[fold_id]["test_indices"]
        ]

        if labels != expected_labels:
            raise ValueError(
                f"Incorrect outer-test labels: {result_path}"
            )

        if len(predictions) != len(labels):
            raise ValueError(
                f"Misaligned predictions and labels: {result_path}"
            )

        if any(
            type(prediction) is not int
            or not 0 <= prediction < dataset.num_classes
            for prediction in predictions
        ):
            raise ValueError(
                f"Invalid predicted class: {result_path}"
            )

        accuracy = sum(
            prediction == label
            for prediction, label in zip(
                predictions,
                labels,
            )
        ) / len(labels)

        if not math.isclose(
            accuracy,
            outer_test["accuracy"],
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Stored accuracy disagrees with predictions: {result_path}"
            )

        for loss in [
            selection["validation_loss"],
            outer_test["loss"],
        ]:
            if not math.isfinite(loss) or loss < 0:
                raise ValueError(
                    f"Invalid recorded loss: {result_path}"
                )

        if not 0 <= selection["validation_accuracy"] <= 1:
            raise ValueError(
                f"Invalid validation accuracy: {result_path}"
            )

        parameters = result["parameters"]

        if not 0 < parameters["trainable"] <= parameters["total"]:
            raise ValueError(
                f"Invalid parameter counts: {result_path}"
            )

        runtime = result["runtime"]
        training_seconds = runtime["training_seconds"]

        if not math.isfinite(training_seconds) or training_seconds < 0:
            raise ValueError(
                f"Invalid training time: {result_path}"
            )

        if not math.isclose(
            runtime["mean_seconds_per_epoch"],
            training_seconds / selection["completed_epochs"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Inconsistent mean epoch time: {result_path}"
            )

        source_commit = result["source_commit"]

        if not isinstance(source_commit, str) or not source_commit:
            raise ValueError(
                f"Invalid source commit: {result_path}"
            )

        results.append(result)

    source_commits = {
        result["source_commit"]
        for result in results
    }

    if len(source_commits) != 1:
        raise ValueError(
            "Outer folds in one result group have different source commits"
        )

    return results


def check_parameter_counts(results, description):
    expected_parameters = results[0]["parameters"]

    if any(
        result["parameters"] != expected_parameters
        for result in results
    ):
        raise ValueError(
            f"Parameter counts differ: {description}"
        )


def check_runtime_conditions(results):
    first_result = results[0]

    for result in results:
        for name in [
            "device",
            "pytorch_version",
            "pyg_version",
            "cuda_version",
        ]:
            if result[name] != first_result[name]:
                raise ValueError(
                    f"Inconsistent runtime conditions: {name}"
                )

        if (
            result["runtime"]["convention"]
            != first_result["runtime"]["convention"]
        ):
            raise ValueError(
                "Inconsistent runtime conventions"
            )

    return first_result