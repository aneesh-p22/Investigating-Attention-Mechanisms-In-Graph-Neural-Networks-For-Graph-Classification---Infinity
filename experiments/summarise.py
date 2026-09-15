import numpy as np

from experiments.ablation import get_variant_settings, head_variants, uniform_variant
from experiments.result_validation import (
    check_runtime_conditions,
    get_partitions,
    load_group,
)
from experiments.train_cv import datasets, model_settings, models, settings
from src.data import load_dataset


def print_source_commits(results):
    print("Source commits:")

    for source_commit in sorted({result["source_commit"] for result in results}):
        print(source_commit)


def print_accuracy_table(groups, label_name, setting_name):
    print("Outer-test accuracy (%):")
    fold_columns = " | ".join(
        f"Outer fold {fold_id}"
        for fold_id in range(settings["num_folds"])
    )
    print(f"Dataset | {label_name} | {fold_columns} | Mean | Sample SD")

    for results in groups:
        current_settings = results[0]["settings"]
        accuracies = np.array(
            [result["outer_test"]["accuracy"] for result in results]
        ) * 100
        fold_values = " | ".join(f"{accuracy:.2f}" for accuracy in accuracies)

        print(
            f"{current_settings['dataset']} | {current_settings[setting_name]} | "
            f"{fold_values} | {accuracies.mean():.2f} | {accuracies.std(ddof=1):.2f}"
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
            f"{current_settings['dataset']} | {current_settings[setting_name]} | "
            f"{parameters['total']} | {parameters['trainable']} | "
            f"{training_seconds:.2f} | {training_seconds / completed_epochs:.4f}"
        )


def get_paired_accuracy_differences(
    baseline_results,
    comparison_results,
    dataset_name,
):
    differences = []

    for baseline_result, comparison_result in zip(baseline_results, comparison_results):
        fold_id = baseline_result["settings"]["fold_id"]

        if comparison_result["settings"]["fold_id"] != fold_id:
            raise ValueError(f"Unmatched outer folds: {dataset_name}")

        if baseline_result["partitions"] != comparison_result["partitions"]:
            raise ValueError(f"Unmatched partitions: {dataset_name}, fold {fold_id}")

        if baseline_result["outer_test"]["labels"] != comparison_result["outer_test"]["labels"]:
            raise ValueError(f"Unmatched test labels: {dataset_name}, fold {fold_id}")

        differences.append(
            (
                comparison_result["outer_test"]["accuracy"]
                - baseline_result["outer_test"]["accuracy"]
            )
            * 100
        )

    return np.array(differences)


def print_gat_gatv2_accuracy_comparison(reference_gat_groups, reference_gatv2_groups):
    print("Outer-test accuracy comparison:")
    print("Differences are GATv2 minus GAT in percentage points.")
    fold_columns = " | ".join(
        f"Outer fold {fold_id}"
        for fold_id in range(settings["num_folds"])
    )
    print(
        f"Dataset | GAT mean | GATv2 mean | {fold_columns} | "
        "Mean difference | Sample SD"
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
        fold_values = " | ".join(f"{difference:.2f}" for difference in differences)

        print(
            f"{dataset_name} | {gat_accuracies.mean():.2f} | {gatv2_accuracies.mean():.2f} | "
            f"{fold_values} | {differences.mean():.2f} | {differences.std(ddof=1):.2f}"
        )


def print_gat_gatv2_parameter_comparison(reference_gat_groups, reference_gatv2_groups):
    print("Parameter counts:")
    print(
        "Dataset | GAT total | GAT trainable | GATv2 total | GATv2 trainable | "
        "GATv2 minus GAT"
    )

    for dataset_name in datasets:
        gat_parameters = reference_gat_groups[dataset_name][0]["parameters"]
        gatv2_parameters = reference_gatv2_groups[dataset_name][0]["parameters"]

        print(
            f"{dataset_name} | {gat_parameters['total']} | {gat_parameters['trainable']} | "
            f"{gatv2_parameters['total']} | {gatv2_parameters['trainable']} | "
            f"{gatv2_parameters['total'] - gat_parameters['total']}"
        )


def print_learned_uniform_accuracy_comparison(reference_gat_groups, uniform_gat_groups):
    print("Outer-test accuracy comparison:")
    print("Differences are learned GAT minus Uniform GAT in percentage points.")
    fold_columns = " | ".join(
        f"Outer fold {fold_id}"
        for fold_id in range(settings["num_folds"])
    )
    print(
        f"Dataset | Learned GAT mean | Uniform GAT mean | {fold_columns} | "
        "Mean difference | Sample SD"
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
        fold_values = " | ".join(f"{difference:.2f}" for difference in differences)

        print(
            f"{dataset_name} | {learned_accuracies.mean():.2f} | "
            f"{uniform_accuracies.mean():.2f} | {fold_values} | "
            f"{differences.mean():.2f} | {differences.std(ddof=1):.2f}"
        )


def print_learned_uniform_parameter_comparison(reference_gat_groups, uniform_gat_groups):
    print("Parameter counts:")
    print(
        "Dataset | Learned total | Learned trainable | Uniform total | "
        "Uniform trainable | Uniform frozen"
    )

    for dataset_name in datasets:
        learned_parameters = reference_gat_groups[dataset_name][0]["parameters"]
        uniform_parameters = uniform_gat_groups[dataset_name][0]["parameters"]

        print(
            f"{dataset_name} | {learned_parameters['total']} | "
            f"{learned_parameters['trainable']} | {uniform_parameters['total']} | "
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

            results = load_group(current_settings, dataset, partitions)
            groups.append(results)

            if model_name == "GAT":
                reference_gat_groups[dataset_name] = results
            elif model_name == "GATv2":
                reference_gatv2_groups[dataset_name] = results

    all_results = [result for group in groups for result in group]
    first_result = check_runtime_conditions(all_results)

    print("Validated reference fits:", len(all_results))
    print("Dataset/model groups:", len(groups))
    print("Outer folds per group:", settings["num_folds"])
    print("Device:", first_result["device"])
    print("PyTorch:", first_result["pytorch_version"])
    print("PyG:", first_result["pyg_version"])
    print("CUDA build:", first_result["cuda_version"])
    print_source_commits(all_results)

    print()
    print_accuracy_table(groups, "Model", "model")

    print()
    print_resource_table(groups, "Model", "model")

    head_groups = []

    for dataset_name in datasets:
        dataset = dataset_information[dataset_name]["dataset"]
        partitions = dataset_information[dataset_name]["partitions"]
        dataset_head_groups = []

        for variant in head_variants:
            current_settings = get_variant_settings(variant)
            current_settings["dataset"] = dataset_name
            dataset_head_groups.append(load_group(current_settings, dataset, partitions))

        dataset_head_groups.append(reference_gat_groups[dataset_name])
        expected_parameters = dataset_head_groups[0][0]["parameters"]

        if any(
            results[0]["parameters"] != expected_parameters
            for results in dataset_head_groups
        ):
            raise ValueError(f"Parameter counts differ across head counts: {dataset_name}")

        head_groups.extend(dataset_head_groups)

    all_head_results = [result for group in head_groups for result in group]
    check_runtime_conditions(all_head_results)
    new_fit_count = len(head_variants) * len(datasets) * settings["num_folds"]

    print()
    print("RQ1 fixed-width GAT head count:")
    print("Validated fits including reused reference:", len(all_head_results))
    print("New fits:", new_fit_count)
    print_source_commits(all_head_results)

    print()
    print_accuracy_table(head_groups, "Heads", "heads")

    print()
    print_resource_table(head_groups, "Heads", "heads")

    rq2_results = []

    for dataset_name in datasets:
        rq2_results.extend(reference_gat_groups[dataset_name])
        rq2_results.extend(reference_gatv2_groups[dataset_name])

    check_runtime_conditions(rq2_results)

    print()
    print("RQ2 matched GAT and GATv2:")
    print("Validated reused reference fits:", len(rq2_results))
    print("New fits: 0")
    print_source_commits(rq2_results)

    print()
    print_gat_gatv2_accuracy_comparison(reference_gat_groups, reference_gatv2_groups)

    print()
    print_gat_gatv2_parameter_comparison(reference_gat_groups, reference_gatv2_groups)

    uniform_gat_groups = {}
    rq3_results = []

    for dataset_name in datasets:
        dataset = dataset_information[dataset_name]["dataset"]
        partitions = dataset_information[dataset_name]["partitions"]
        current_settings = get_variant_settings(uniform_variant)
        current_settings["dataset"] = dataset_name
        uniform_results = load_group(current_settings, dataset, partitions)
        learned_results = reference_gat_groups[dataset_name]
        learned_parameters = learned_results[0]["parameters"]
        uniform_parameters = uniform_results[0]["parameters"]

        if uniform_parameters["total"] != learned_parameters["total"]:
            raise ValueError(f"Total parameters differ for uniform control: {dataset_name}")

        if learned_parameters["trainable"] != learned_parameters["total"]:
            raise ValueError(f"Reference GAT has frozen parameters: {dataset_name}")

        if not 0 < uniform_parameters["trainable"] < uniform_parameters["total"]:
            raise ValueError(f"Uniform GAT does not have frozen parameters: {dataset_name}")

        uniform_gat_groups[dataset_name] = uniform_results
        rq3_results.extend(learned_results)
        rq3_results.extend(uniform_results)

    check_runtime_conditions(rq3_results)

    print()
    print("RQ3 learned versus uniform-attention retraining:")
    print("Validated fits including reused reference:", len(rq3_results))
    print("New fits:", len(datasets) * settings["num_folds"])
    print_source_commits(rq3_results)

    print()
    print_learned_uniform_accuracy_comparison(reference_gat_groups, uniform_gat_groups)

    print()
    print_learned_uniform_parameter_comparison(reference_gat_groups, uniform_gat_groups)


if __name__ == "__main__":
    main()