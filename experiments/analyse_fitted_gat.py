import math

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from experiments.result_validation import (
    get_partitions,
    load_group,
)
from experiments.train_cv import (
    datasets,
    model_settings,
    settings,
)
from src.data import load_dataset
from src.evaluation import evaluate
from src.models.factory import build_model
from src.recording import (
    get_source_commit,
    prepare_result_path,
    save_result,
)


result_path = "results/rq4_fitted_gat_attention.json"

attention_parameters = {
    "conv1.att_src",
    "conv1.att_dst",
    "conv2.att_src",
    "conv2.att_dst",
}


def check_identical_models(
    learned_model,
    intervention_model,
):
    learned_state = learned_model.state_dict()
    intervention_state = intervention_model.state_dict()

    if learned_state.keys() != intervention_state.keys():
        raise ValueError("Model copies have different state keys")

    for name in learned_state:
        if not torch.equal(
            learned_state[name],
            intervention_state[name],
        ):
            raise ValueError(
                f"Model copies differ before intervention: {name}"
            )


def apply_uniform_intervention(
    learned_model,
    intervention_model,
):
    with torch.no_grad():
        intervention_model.conv1.att_src.zero_()
        intervention_model.conv1.att_dst.zero_()
        intervention_model.conv2.att_src.zero_()
        intervention_model.conv2.att_dst.zero_()

    learned_state = learned_model.state_dict()
    intervention_state = intervention_model.state_dict()

    for name in learned_state:
        if name in attention_parameters:
            if not torch.all(
                intervention_state[name] == 0
            ).item():
                raise ValueError(
                    f"Attention scorer is not zero: {name}"
                )

        elif not torch.equal(
            learned_state[name],
            intervention_state[name],
        ):
            raise ValueError(
                f"Intervention changed another parameter: {name}"
            )


def check_reference_evaluation(
    result,
    loss,
    accuracy,
    predictions,
    labels,
):
    stored_test = result["outer_test"]

    if labels != stored_test["labels"]:
        raise ValueError(
            "Recomputed labels differ from the reference result"
        )

    if predictions != stored_test["predictions"]:
        raise ValueError(
            "Recomputed predictions differ from the reference result"
        )

    if not math.isclose(
        accuracy,
        stored_test["accuracy"],
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Recomputed accuracy differs from the reference result"
        )

    if not math.isclose(
        loss,
        stored_test["loss"],
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Recomputed loss differs from the reference result"
        )


def get_layer_departure(
    edge_index,
    attention,
    num_nodes,
):
    if attention.dim() == 1:
        attention = attention.unsqueeze(1)

    if edge_index.shape[1] != attention.shape[0]:
        raise ValueError(
            "Attention coefficients do not match returned edges"
        )

    if (
        not torch.isfinite(attention).all().item()
        or torch.any(attention < 0).item()
    ):
        raise ValueError(
            "Invalid attention coefficients"
        )

    receivers = edge_index[1]
    departures = []

    eligible_receivers = 0
    single_entry_receivers = 0

    for receiver in range(num_nodes):
        incoming = attention[
            receivers == receiver
        ]

        if incoming.shape[0] == 0:
            raise ValueError(
                "Receiving node has no attention entries"
            )

        coefficient_sums = incoming.sum(dim=0)

        if not torch.allclose(
            coefficient_sums,
            torch.ones_like(coefficient_sums),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(
                "Incoming attention coefficients do not sum to one"
            )

        incoming_entries = incoming.shape[0]

        if incoming_entries == 1:
            single_entry_receivers += 1
            continue

        entropy_terms = torch.zeros_like(incoming)
        positive = incoming > 0

        entropy_terms[positive] = (
            incoming[positive]
            * torch.log(incoming[positive])
        )

        entropy = (
            -entropy_terms.sum(dim=0)
            / math.log(incoming_entries)
        )

        departure = 1.0 - entropy

        if (
            departure.min().item() < -1e-6
            or departure.max().item() > 1.0 + 1e-6
        ):
            raise ValueError(
                "Attention departure is outside the expected range"
            )

        departures.append(
            departure.mean().item()
        )

        eligible_receivers += 1

    if not departures:
        raise ValueError(
            "Graph has no receiver with multiple attention entries"
        )

    return {
        "departure": float(
            np.mean(departures)
        ),
        "eligible_receivers": eligible_receivers,
        "single_entry_receivers": single_entry_receivers,
    }


def get_graph_attention(
    model,
    graph,
):
    model.eval()

    with torch.no_grad():
        x, (
            conv1_edges,
            conv1_attention,
        ) = model.conv1(
            graph.x,
            graph.edge_index,
            return_attention_weights=True,
        )

        conv1 = get_layer_departure(
            conv1_edges,
            conv1_attention,
            graph.num_nodes,
        )

        x = F.relu(x)

        _, (
            conv2_edges,
            conv2_attention,
        ) = model.conv2(
            x,
            graph.edge_index,
            return_attention_weights=True,
        )

        conv2 = get_layer_departure(
            conv2_edges,
            conv2_attention,
            graph.num_nodes,
        )

    return {
        "conv1": conv1,
        "conv2": conv2,
    }


def analyse_fold(
    result,
    current_settings,
    dataset,
    device,
):
    test_indices = result["partitions"]["test_indices"]

    learned_model = build_model(
        current_settings,
        dataset.num_node_features,
        dataset.num_classes,
    ).to(device)

    intervention_model = build_model(
        current_settings,
        dataset.num_node_features,
        dataset.num_classes,
    ).to(device)

    state = torch.load(
        result["model_state_path"],
        map_location=device,
        weights_only=True,
    )

    learned_model.load_state_dict(state)
    intervention_model.load_state_dict(state)

    check_identical_models(
        learned_model,
        intervention_model,
    )

    test_loader = DataLoader(
        dataset[test_indices],
        batch_size=current_settings["batch_size"],
        shuffle=False,
    )

    (
        learned_loss,
        learned_accuracy,
        learned_predictions,
        learned_labels,
    ) = evaluate(
        learned_model,
        test_loader,
        device,
    )

    check_reference_evaluation(
        result,
        learned_loss,
        learned_accuracy,
        learned_predictions,
        learned_labels,
    )

    graph_attention = []

    for graph_id in test_indices:
        graph = dataset[graph_id].to(device)

        attention = get_graph_attention(
            learned_model,
            graph,
        )

        graph_attention.append(
            {
                "graph_id": graph_id,
                "conv1": attention["conv1"],
                "conv2": attention["conv2"],
            }
        )

    conv1_fold_departure = np.mean(
        [
            graph["conv1"]["departure"]
            for graph in graph_attention
        ]
    )

    conv2_fold_departure = np.mean(
        [
            graph["conv2"]["departure"]
            for graph in graph_attention
        ]
    )

    apply_uniform_intervention(
        learned_model,
        intervention_model,
    )

    (
        uniform_loss,
        uniform_accuracy,
        uniform_predictions,
        uniform_labels,
    ) = evaluate(
        intervention_model,
        test_loader,
        device,
    )

    if uniform_labels != learned_labels:
        raise ValueError(
            "Intervention changed label alignment"
        )

    graphs = []

    for (
        graph_id,
        label,
        learned_prediction,
        uniform_prediction,
        attention,
    ) in zip(
        test_indices,
        learned_labels,
        learned_predictions,
        uniform_predictions,
        graph_attention,
    ):
        if graph_id != attention["graph_id"]:
            raise ValueError(
                "Attention results are not aligned with graph IDs"
            )

        graphs.append(
            {
                "graph_id": graph_id,
                "label": label,
                "learned_prediction": learned_prediction,
                "uniform_prediction": uniform_prediction,
                "prediction_changed": (
                    learned_prediction
                    != uniform_prediction
                ),
                "conv1_departure": attention[
                    "conv1"
                ]["departure"],
                "conv2_departure": attention[
                    "conv2"
                ]["departure"],
                "conv1_eligible_receivers": attention[
                    "conv1"
                ]["eligible_receivers"],
                "conv1_single_entry_receivers": attention[
                    "conv1"
                ]["single_entry_receivers"],
                "conv2_eligible_receivers": attention[
                    "conv2"
                ]["eligible_receivers"],
                "conv2_single_entry_receivers": attention[
                    "conv2"
                ]["single_entry_receivers"],
            }
        )

    prediction_flips = sum(
        graph["prediction_changed"]
        for graph in graphs
    )

    return {
        "fold_id": result["settings"]["fold_id"],
        "model_state_path": result["model_state_path"],
        "selected_epoch": result["selection"]["selected_epoch"],
        "source_commit": result["source_commit"],
        "attention": {
            "conv1_mean_departure": float(
                conv1_fold_departure
            ),
            "conv2_mean_departure": float(
                conv2_fold_departure
            ),
        },
        "intervention": {
            "learned_loss": learned_loss,
            "uniform_loss": uniform_loss,
            "delta_loss": (
                uniform_loss - learned_loss
            ),
            "learned_accuracy": learned_accuracy,
            "uniform_accuracy": uniform_accuracy,
            "delta_accuracy": (
                uniform_accuracy - learned_accuracy
            ),
            "prediction_flip_rate": (
                prediction_flips / len(graphs)
            ),
        },
        "graphs": graphs,
    }


def get_summary(values):
    values = np.array(values)

    return {
        "fold_values": values.tolist(),
        "mean": float(values.mean()),
        "sample_sd": float(
            values.std(ddof=1)
        ),
    }


def summarise_dataset(folds):
    conv1_departure = [
        fold["attention"]["conv1_mean_departure"]
        for fold in folds
    ]

    conv2_departure = [
        fold["attention"]["conv2_mean_departure"]
        for fold in folds
    ]

    delta_loss = [
        fold["intervention"]["delta_loss"]
        for fold in folds
    ]

    delta_accuracy = [
        fold["intervention"]["delta_accuracy"]
        for fold in folds
    ]

    prediction_flip_rate = [
        fold["intervention"]["prediction_flip_rate"]
        for fold in folds
    ]

    return {
        "conv1_departure": get_summary(
            conv1_departure
        ),
        "conv2_departure": get_summary(
            conv2_departure
        ),
        "delta_loss": get_summary(
            delta_loss
        ),
        "delta_accuracy": get_summary(
            delta_accuracy
        ),
        "prediction_flip_rate": get_summary(
            prediction_flip_rate
        ),
        "conv1_eligible_receivers": sum(
            graph["conv1_eligible_receivers"]
            for fold in folds
            for graph in fold["graphs"]
        ),
        "conv1_single_entry_receivers": sum(
            graph["conv1_single_entry_receivers"]
            for fold in folds
            for graph in fold["graphs"]
        ),
        "conv2_eligible_receivers": sum(
            graph["conv2_eligible_receivers"]
            for fold in folds
            for graph in fold["graphs"]
        ),
        "conv2_single_entry_receivers": sum(
            graph["conv2_single_entry_receivers"]
            for fold in folds
            for graph in fold["graphs"]
        ),
    }


def print_summary(dataset_results):
    print("Attention departure from uniform weighting:")

    for dataset_name in datasets:
        summary = dataset_results[
            dataset_name
        ]["summary"]

        print()
        print("Dataset:", dataset_name)

        print(
            "Conv1 fold departures:",
            [
                round(value, 4)
                for value in summary[
                    "conv1_departure"
                ]["fold_values"]
            ],
        )

        print(
            "Conv1 mean:",
            f"{summary['conv1_departure']['mean']:.4f}",
        )

        print(
            "Conv1 sample SD:",
            f"{summary['conv1_departure']['sample_sd']:.4f}",
        )

        print(
            "Conv2 fold departures:",
            [
                round(value, 4)
                for value in summary[
                    "conv2_departure"
                ]["fold_values"]
            ],
        )

        print(
            "Conv2 mean:",
            f"{summary['conv2_departure']['mean']:.4f}",
        )

        print(
            "Conv2 sample SD:",
            f"{summary['conv2_departure']['sample_sd']:.4f}",
        )

    print()
    print("Uniform-attention intervention:")
    print("Delta loss is uniform minus learned.")
    print("Delta accuracy is uniform minus learned.")

    for dataset_name in datasets:
        summary = dataset_results[
            dataset_name
        ]["summary"]

        print()
        print("Dataset:", dataset_name)

        print(
            "Delta loss folds:",
            [
                round(value, 4)
                for value in summary[
                    "delta_loss"
                ]["fold_values"]
            ],
        )

        print(
            "Delta loss mean:",
            f"{summary['delta_loss']['mean']:.4f}",
        )

        print(
            "Delta accuracy folds (pp):",
            [
                round(value * 100, 2)
                for value in summary[
                    "delta_accuracy"
                ]["fold_values"]
            ],
        )

        print(
            "Delta accuracy mean (pp):",
            f"{summary['delta_accuracy']['mean'] * 100:.2f}",
        )

        print(
            "Prediction flip folds (%):",
            [
                round(value * 100, 2)
                for value in summary[
                    "prediction_flip_rate"
                ]["fold_values"]
            ],
        )

        print(
            "Prediction flip mean (%):",
            f"{summary['prediction_flip_rate']['mean'] * 100:.2f}",
        )


def main():
    prepare_result_path(result_path)

    source_commit = get_source_commit()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(
            device
        )
    else:
        device_name = "CPU"

    dataset_results = {}
    reference_source_commits = set()
    selected_state_count = 0

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        partitions = get_partitions(dataset)

        current_settings = settings.copy()
        current_settings.update(
            model_settings["GAT"]
        )
        current_settings["dataset"] = dataset_name
        current_settings["model"] = "GAT"

        reference_results = load_group(
            current_settings,
            dataset,
            partitions,
        )

        folds = []

        for result in reference_results:
            folds.append(
                analyse_fold(
                    result,
                    current_settings,
                    dataset,
                    device,
                )
            )

            reference_source_commits.add(
                result["source_commit"]
            )

            selected_state_count += 1

        dataset_results[dataset_name] = {
            "folds": folds,
            "summary": summarise_dataset(
                folds
            ),
        }

    expected_state_count = (
        len(datasets)
        * settings["num_folds"]
    )

    if selected_state_count != expected_state_count:
        raise ValueError(
            "Unexpected number of selected reference states"
        )

    if len(reference_source_commits) != 1:
        raise ValueError(
            "Reference GAT states have different source commits"
        )

    result = {
        "purpose": "rq4_fitted_gat_attention",
        "source_commit": source_commit,
        "reference_source_commit": next(
            iter(reference_source_commits)
        ),
        "device": {
            "type": device.type,
            "name": device_name,
        },
        "settings": {
            "model": "GAT",
            "heads": model_settings["GAT"]["heads"],
            "num_folds": settings["num_folds"],
            "attention_measure": (
                "one minus normalised attention entropy"
            ),
            "single_entry_receivers": (
                "excluded from entropy and retained as counts"
            ),
            "intervention": (
                "attention scoring tensors zeroed without retraining"
            ),
        },
        "selected_reference_states": selected_state_count,
        "new_optimisation_fits": 0,
        "datasets": dataset_results,
    }

    save_result(
        result_path,
        result,
    )

    print("RQ4 fitted reference-GAT analysis:")
    print(
        "Selected reference states:",
        selected_state_count,
    )
    print("New optimisation fits: 0")
    print(
        "Analysis source commit:",
        source_commit,
    )
    print(
        "Reference source commit:",
        result["reference_source_commit"],
    )

    print()
    print_summary(
        dataset_results
    )

    print()
    print(
        "Saved result:",
        result_path,
    )


if __name__ == "__main__":
    main()