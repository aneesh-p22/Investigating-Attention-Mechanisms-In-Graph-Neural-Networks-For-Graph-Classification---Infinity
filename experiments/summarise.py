import json
import math
import os

import numpy as np

from experiments.ablation import (
    get_variant_settings,
    head_variants,
    uniform_variant,
)
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
        f"{current_settings['dataset'].lower()}"
    )

    if current_settings["variant"] is not None:
        prefix += f"_{current_settings['variant']}"

    prefix += "_fold"

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

        if not math.isclose(
            accuracy,
            outer_test["accuracy"],
            rel_tol=0,
            abs_tol=1e-12,
        ):
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


def check_parameter_counts(results, description):
    if any(
        result["parameters"] != results[0]["parameters"]
        for result in results
    ):
        raise ValueError(f"Parameter counts differ: {description}")


def check_runtime_conditions(results):
    first_result = results[0]

    for result in results:
        for name in ["device", "pytorch_version", "pyg_version", "cuda_version"]:
            if result[name] != first_result[name]:
                raise ValueError(f"Inconsistent runtime conditions: {name}")

        if result["runtime"]["convention"] != first_result["runtime"]["convention"]:
            raise ValueError("Inconsistent runtime conventions")

    return first_result


def print_accuracy_table(groups, label_name, setting_name):
    print("Outer-test accuracy (%):")

    fold_columns = " | ".join(
        f"Outer fold {fold_id}"
        for fold_id in range(settings["num_folds"])
    )

    print(
        f"Dataset | {label_name} | "
        f"{fold_columns} | Mean | Sample SD"
    )

    for results in groups:
        current_settings = results[0]["settings"]
        accuracies = np.array(
            [result["outer_test"]["accuracy"] for result in results]
        ) * 100
        fold_values = " | ".join(
            f"{accuracy:.2f}"
            for accuracy in accuracies
        )

        print(
            f"{current_settings['dataset']} | "
            f"{current_settings[setting_name]} | "
            f"{fold_values} | "
            f"{accuracies.mean():.2f} | "
            f"{accuracies.std(ddof=1):.2f}"
        )


def print_resource_table(groups, label_name, setting_name):
    print("Parameters and training time:")
    print("Training passes only; time totals cover all outer folds.")
    print(
        f"Dataset | {label_name} | Total parameters | Trainable parameters | "
        "Total time (s) | Mean epoch time (s)"
    )

    for results in groups:
        current_settings = results[0]["settings"]
        parameters = results[0]["parameters"]
        training_seconds = sum(
            result["runtime"]["training_seconds"]
            for result in results
        )
        completed_epochs = sum(
            result["selection"]["completed_epochs"]
            for result in results
        )

        print(
            f"{current_settings['dataset']} | "
            f"{current_settings[setting_name]} | "
            f"{parameters['total']} | "
            f"{parameters['trainable']} | "
            f"{training_seconds:.2f} | "
            f"{training_seconds / completed_epochs:.4f}"
        )


def get_paired_accuracy_differences(
    baseline_results,
    comparison_results,
    dataset_name,
):
    differences = []

    for baseline_result, comparison_result in zip(
        baseline_results,
        comparison_results,
    ):
        baseline_fold = baseline_result["settings"]["fold_id"]
        comparison_fold = comparison_result["settings"]["fold_id"]

        if baseline_fold != comparison_fold:
            raise ValueError(f"Unmatched outer folds: {dataset_name}")

        if baseline_result["partitions"] != comparison_result["partitions"]:
            raise ValueError(
                f"Unmatched partitions: {dataset_name}, fold {baseline_fold}"
            )

        if (
            baseline_result["outer_test"]["labels"]
            != comparison_result["outer_test"]["labels"]
        ):
            raise ValueError(
                f"Unmatched test labels: {dataset_name}, fold {baseline_fold}"
            )

        difference = (
            comparison_result["outer_test"]["accuracy"]
            - baseline_result["outer_test"]["accuracy"]
        ) * 100

        differences.append(difference)

    return np.array(differences)


def print_gat_gatv2_accuracy_comparison(
    reference_gat_groups,
    reference_gatv2_groups,
):
    print("Outer-test accuracy comparison:")
    print("Differences are GATv2 minus GAT in percentage points.")

    fold_columns = " | ".join(
        f"Outer fold {fold_id}"
        for fold_id in range(settings["num_folds"])
    )

    print(
        f"Dataset | GAT mean | GATv2 mean | "
        f"{fold_columns} | Mean difference | Sample SD"
    )

    for dataset_name in datasets:
        gat_results = reference_gat_groups[dataset_name]
        gatv2_results = reference_gatv2_groups[dataset_name]

        gat_accuracies = np.array(
            [result["outer_test"]["accuracy"] for result in gat_results]
        ) * 100
        gatv2_accuracies = np.array(
            [result["outer_test"]["accuracy"] for result in gatv2_results]
        ) * 100

        differences = get_paired_accuracy_differences(
            gat_results,
            gatv2_results,
            dataset_name,
        )

        fold_values = " | ".join(
            f"{difference:.2f}"
            for difference in differences
        )

        print(
            f"{dataset_name} | "
            f"{gat_accuracies.mean():.2f} | "
            f"{gatv2_accuracies.mean():.2f} | "
            f"{fold_values} | "
            f"{differences.mean():.2f} | "
            f"{differences.std(ddof=1):.2f}"
        )


def print_gat_gatv2_parameter_comparison(
    reference_gat_groups,
    reference_gatv2_groups,
):
    print("Parameter counts:")
    print(
        "Dataset | GAT total | GAT trainable | "
        "GATv2 total | GATv2 trainable | GATv2 minus GAT"
    )

    for dataset_name in datasets:
        gat_parameters = reference_gat_groups[dataset_name][0]["parameters"]
        gatv2_parameters = reference_gatv2_groups[dataset_name][0]["parameters"]

        print(
            f"{dataset_name} | "
            f"{gat_parameters['total']} | "
            f"{gat_parameters['trainable']} | "
            f"{gatv2_parameters['total']} | "
            f"{gatv2_parameters['trainable']} | "
            f"{gatv2_parameters['total'] - gat_parameters['total']}"
        )


def print_learned_uniform_accuracy_comparison(
    reference_gat_groups,
    uniform_gat_groups,
):
    print("Outer-test accuracy comparison:")
    print("Differences are learned GAT minus Uniform GAT in percentage points.")

    fold_columns = " | ".join(
        f"Outer fold {fold_id}"
        for fold_id in range(settings["num_folds"])
    )

    print(
        f"Dataset | Learned GAT mean | Uniform GAT mean | "
        f"{fold_columns} | Mean difference | Sample SD"
    )

    for dataset_name in datasets:
        learned_results = reference_gat_groups[dataset_name]
        uniform_results = uniform_gat_groups[dataset_name]

        learned_accuracies = np.array(
            [result["outer_test"]["accuracy"] for result in learned_results]
        ) * 100
        uniform_accuracies = np.array(
            [result["outer_test"]["accuracy"] for result in uniform_results]
        ) * 100

        differences = get_paired_accuracy_differences(
            uniform_results,
            learned_results,
            dataset_name,
        )

        fold_values = " | ".join(
            f"{difference:.2f}"
            for difference in differences
        )

        print(
            f"{dataset_name} | "
            f"{learned_accuracies.mean():.2f} | "
            f"{uniform_accuracies.mean():.2f} | "
            f"{fold_values} | "
            f"{differences.mean():.2f} | "
            f"{differences.std(ddof=1):.2f}"
        )


def print_learned_uniform_parameter_comparison(
    reference_gat_groups,
    uniform_gat_groups,
):
    print("Parameter counts:")
    print(
        "Dataset | Learned total | Learned trainable | "
        "Uniform total | Uniform trainable | Uniform frozen"
    )

    for dataset_name in datasets:
        learned_parameters = (
            reference_gat_groups[dataset_name][0]["parameters"]
        )
        uniform_parameters = (
            uniform_gat_groups[dataset_name][0]["parameters"]
        )

        print(
            f"{dataset_name} | "
            f"{learned_parameters['total']} | "
            f"{learned_parameters['trainable']} | "
            f"{uniform_parameters['total']} | "
            f"{uniform_parameters['trainable']} | "
            f"{uniform_parameters['total'] - uniform_parameters['trainable']}"
        )


def main():
    groups = []
    reference_gat_groups = {}
    reference_gatv2_groups = {}
    dataset_information = {}

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        partitions = get_partitions(dataset)

        dataset_information[dataset_name] = {
            "dataset": dataset,
            "partitions": partitions,
        }

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

            check_parameter_counts(
                results,
                f"{dataset_name}, {model_name}",
            )

            groups.append(results)

            if model_name == "GAT":
                reference_gat_groups[dataset_name] = results

            if model_name == "GATv2":
                reference_gatv2_groups[dataset_name] = results

    all_results = [
        result
        for group in groups
        for result in group
    ]
    first_result = check_runtime_conditions(all_results)

    print("Validated reference fits:", len(all_results))
    print("Dataset/model groups:", len(groups))
    print("Outer folds per group:", settings["num_folds"])
    print("Device:", first_result["device"])
    print("PyTorch:", first_result["pytorch_version"])
    print("PyG:", first_result["pyg_version"])
    print("CUDA build:", first_result["cuda_version"])
    print("Source commits:")

    for source_commit in sorted(
        {result["source_commit"] for result in all_results}
    ):
        print(source_commit)

    print()
    print_accuracy_table(
        groups,
        "Model",
        "model",
    )

    print()
    print_resource_table(
        groups,
        "Model",
        "model",
    )

    head_groups = []

    for dataset_name in datasets:
        dataset = dataset_information[dataset_name]["dataset"]
        partitions = dataset_information[dataset_name]["partitions"]
        dataset_head_groups = []

        for variant in head_variants:
            current_settings = get_variant_settings(variant)
            current_settings["dataset"] = dataset_name

            results = load_group(
                current_settings,
                dataset,
                partitions,
            )

            check_parameter_counts(
                results,
                f"{dataset_name}, {variant['name']}",
            )

            dataset_head_groups.append(results)

        dataset_head_groups.append(
            reference_gat_groups[dataset_name]
        )

        expected_parameters = dataset_head_groups[0][0]["parameters"]

        if any(
            results[0]["parameters"] != expected_parameters
            for results in dataset_head_groups
        ):
            raise ValueError(
                f"Parameter counts differ across head counts: {dataset_name}"
            )

        head_groups.extend(dataset_head_groups)

    all_head_results = [
        result
        for group in head_groups
        for result in group
    ]
    check_runtime_conditions(all_head_results)

    new_fit_count = (
        len(head_variants)
        * len(datasets)
        * settings["num_folds"]
    )

    print()
    print("RQ1 fixed-width GAT head count:")
    print(
        "Validated fits including reused reference:",
        len(all_head_results),
    )
    print("New fits:", new_fit_count)
    print("Source commits:")

    for source_commit in sorted(
        {result["source_commit"] for result in all_head_results}
    ):
        print(source_commit)

    print()
    print_accuracy_table(
        head_groups,
        "Heads",
        "heads",
    )

    print()
    print_resource_table(
        head_groups,
        "Heads",
        "heads",
    )

    rq2_results = []

    for dataset_name in datasets:
        rq2_results.extend(reference_gat_groups[dataset_name])
        rq2_results.extend(reference_gatv2_groups[dataset_name])

    check_runtime_conditions(rq2_results)

    print()
    print("RQ2 matched GAT and GATv2:")
    print("Validated reused reference fits:", len(rq2_results))
    print("New fits: 0")
    print("Source commits:")

    for source_commit in sorted(
        {result["source_commit"] for result in rq2_results}
    ):
        print(source_commit)

    print()
    print_gat_gatv2_accuracy_comparison(
        reference_gat_groups,
        reference_gatv2_groups,
    )

    print()
    print_gat_gatv2_parameter_comparison(
        reference_gat_groups,
        reference_gatv2_groups,
    )

    uniform_gat_groups = {}
    rq3_results = []

    for dataset_name in datasets:
        dataset = dataset_information[dataset_name]["dataset"]
        partitions = dataset_information[dataset_name]["partitions"]

        current_settings = get_variant_settings(uniform_variant)
        current_settings["dataset"] = dataset_name

        uniform_results = load_group(
            current_settings,
            dataset,
            partitions,
        )

        check_parameter_counts(
            uniform_results,
            f"{dataset_name}, {uniform_variant['name']}",
        )

        learned_results = reference_gat_groups[dataset_name]
        learned_parameters = learned_results[0]["parameters"]
        uniform_parameters = uniform_results[0]["parameters"]

        if uniform_parameters["total"] != learned_parameters["total"]:
            raise ValueError(
                f"Total parameters differ for uniform control: {dataset_name}"
            )

        if learned_parameters["trainable"] != learned_parameters["total"]:
            raise ValueError(
                f"Reference GAT has frozen parameters: {dataset_name}"
            )

        if not 0 < uniform_parameters["trainable"] < uniform_parameters["total"]:
            raise ValueError(
                f"Uniform GAT does not have frozen parameters: {dataset_name}"
            )

        uniform_gat_groups[dataset_name] = uniform_results

        rq3_results.extend(learned_results)
        rq3_results.extend(uniform_results)

    check_runtime_conditions(rq3_results)

    print()
    print("RQ3 learned versus uniform-attention retraining:")
    print("Validated fits including reused reference:", len(rq3_results))
    print(
        "New fits:",
        len(datasets) * settings["num_folds"],
    )
    print("Source commits:")

    for source_commit in sorted(
        {result["source_commit"] for result in rq3_results}
    ):
        print(source_commit)

    print()
    print_learned_uniform_accuracy_comparison(
        reference_gat_groups,
        uniform_gat_groups,
    )

    print()
    print_learned_uniform_parameter_comparison(
        reference_gat_groups,
        uniform_gat_groups,
    )


if __name__ == "__main__":
    main()