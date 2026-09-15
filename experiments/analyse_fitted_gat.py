import math

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from experiments.result_validation import get_partitions, load_group
from experiments.train_cv import datasets, model_settings, settings
from src.data import load_dataset
from src.evaluation import evaluate
from src.models.factory import build_model
from src.recording import get_source_commit, prepare_result_path, save_result


result_path = "results/rq4_fitted_gat_attention.json"

attention_parameters = {
    "conv1.att_src",
    "conv1.att_dst",
    "conv2.att_src",
    "conv2.att_dst",
}


def check_reference_evaluation(result, loss, accuracy, predictions, labels):
    fold_id = result["settings"]["fold_id"]
    stored_test = result["outer_test"]

    if labels != stored_test["labels"]:
        raise ValueError(f"Recomputed labels differ in outer fold {fold_id}")

    if predictions != stored_test["predictions"]:
        raise ValueError(f"Recomputed predictions differ in outer fold {fold_id}")

    if not math.isclose(accuracy, stored_test["accuracy"], rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Recomputed accuracy differs in outer fold {fold_id}")

    if not math.isclose(loss, stored_test["loss"], rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Recomputed loss differs in outer fold {fold_id}")


def apply_uniform_intervention(learned_model, intervention_model):
    learned_state = learned_model.state_dict()
    intervention_state = intervention_model.state_dict()

    if learned_state.keys() != intervention_state.keys():
        raise ValueError("Model copies have different state keys")

    for name in learned_state:
        if not torch.equal(learned_state[name], intervention_state[name]):
            raise ValueError(f"Model copies differ before intervention: {name}")

    with torch.no_grad():
        intervention_model.conv1.att_src.zero_()
        intervention_model.conv1.att_dst.zero_()
        intervention_model.conv2.att_src.zero_()
        intervention_model.conv2.att_dst.zero_()

    intervention_state = intervention_model.state_dict()

    for name in learned_state:
        if name in attention_parameters:
            if not torch.all(intervention_state[name] == 0).item():
                raise ValueError(f"Attention scorer is not zero after intervention: {name}")
        elif not torch.equal(learned_state[name], intervention_state[name]):
            raise ValueError(f"Intervention changed another state tensor: {name}")


def get_layer_departure(edge_index, attention, num_nodes):
    if attention.dim() == 1:
        attention = attention.unsqueeze(1)

    if edge_index.shape[1] != attention.shape[0]:
        raise ValueError("Attention coefficients do not match returned edges")

    if not torch.isfinite(attention).all().item() or torch.any(attention < 0).item():
        raise ValueError("Invalid attention coefficients")

    receivers = edge_index[1]
    departures = []
    eligible_receivers = 0
    single_entry_receivers = 0

    for receiver in range(num_nodes):
        incoming = attention[receivers == receiver]

        if incoming.shape[0] == 0:
            raise ValueError("Receiving node has no attention entries")

        coefficient_sums = incoming.sum(dim=0)

        if not torch.allclose(
            coefficient_sums,
            torch.ones_like(coefficient_sums),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("Incoming attention coefficients do not sum to one")

        incoming_entries = incoming.shape[0]

        if incoming_entries == 1:
            single_entry_receivers += 1
            continue

        entropy_terms = torch.zeros_like(incoming)
        positive = incoming > 0
        entropy_terms[positive] = incoming[positive] * torch.log(incoming[positive])
        entropy = -entropy_terms.sum(dim=0) / math.log(incoming_entries)
        departure = 1.0 - entropy

        if departure.min().item() < -1e-6 or departure.max().item() > 1.0 + 1e-6:
            raise ValueError("Attention departure is outside the expected range")

        departures.append(departure.mean().item())
        eligible_receivers += 1

    if not departures:
        raise ValueError("Graph has no receiver with multiple attention entries")

    return {
        "departure": float(np.mean(departures)),
        "eligible_receivers": eligible_receivers,
        "single_entry_receivers": single_entry_receivers,
    }


def get_graph_attention(model, graph):
    model.eval()

    with torch.no_grad():
        x, (conv1_edges, conv1_attention) = model.conv1(
            graph.x,
            graph.edge_index,
            return_attention_weights=True,
        )
        conv1 = get_layer_departure(conv1_edges, conv1_attention, graph.num_nodes)

        x = F.relu(x)
        _, (conv2_edges, conv2_attention) = model.conv2(
            x,
            graph.edge_index,
            return_attention_weights=True,
        )
        conv2 = get_layer_departure(conv2_edges, conv2_attention, graph.num_nodes)

    return conv1, conv2


def analyse_fold(result, current_settings, dataset, device):
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
    ) = evaluate(learned_model, test_loader, device)

    check_reference_evaluation(
        result,
        learned_loss,
        learned_accuracy,
        learned_predictions,
        learned_labels,
    )

    apply_uniform_intervention(learned_model, intervention_model)

    (
        intervention_loss,
        intervention_accuracy,
        intervention_predictions,
        intervention_labels,
    ) = evaluate(intervention_model, test_loader, device)

    if intervention_labels != learned_labels:
        raise ValueError("Intervention changed outer-test label alignment")

    graphs = []
    conv1_departures = []
    conv2_departures = []
    prediction_flips = 0

    for graph_id, label, learned_prediction, intervention_prediction in zip(
        test_indices,
        learned_labels,
        learned_predictions,
        intervention_predictions,
    ):
        graph = dataset[graph_id].to(device)
        conv1, conv2 = get_graph_attention(learned_model, graph)

        conv1_departures.append(conv1["departure"])
        conv2_departures.append(conv2["departure"])
        prediction_flips += learned_prediction != intervention_prediction

        graphs.append(
            {
                "graph_id": graph_id,
                "label": label,
                "learned_prediction": learned_prediction,
                "intervention_prediction": intervention_prediction,
                "conv1_departure": conv1["departure"],
                "conv2_departure": conv2["departure"],
                "conv1_eligible_receivers": conv1["eligible_receivers"],
                "conv1_single_entry_receivers": conv1["single_entry_receivers"],
                "conv2_eligible_receivers": conv2["eligible_receivers"],
                "conv2_single_entry_receivers": conv2["single_entry_receivers"],
            }
        )

    return {
        "fold_id": result["settings"]["fold_id"],
        "model_state_path": result["model_state_path"],
        "selected_epoch": result["selection"]["selected_epoch"],
        "attention": {
            "conv1_mean_departure": float(np.mean(conv1_departures)),
            "conv2_mean_departure": float(np.mean(conv2_departures)),
        },
        "intervention": {
            "learned_loss": learned_loss,
            "intervention_loss": intervention_loss,
            "delta_loss": intervention_loss - learned_loss,
            "learned_accuracy": learned_accuracy,
            "intervention_accuracy": intervention_accuracy,
            "delta_accuracy": intervention_accuracy - learned_accuracy,
            "prediction_flip_rate": prediction_flips / len(graphs),
        },
        "graphs": graphs,
    }


def summarise_values(values):
    values = np.array(values)

    return {
        "fold_values": values.tolist(),
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
    }


def summarise_dataset(folds):
    summary = {
        "conv1_departure": summarise_values(
            [fold["attention"]["conv1_mean_departure"] for fold in folds]
        ),
        "conv2_departure": summarise_values(
            [fold["attention"]["conv2_mean_departure"] for fold in folds]
        ),
        "delta_loss": summarise_values(
            [fold["intervention"]["delta_loss"] for fold in folds]
        ),
        "delta_accuracy": summarise_values(
            [fold["intervention"]["delta_accuracy"] for fold in folds]
        ),
        "prediction_flip_rate": summarise_values(
            [fold["intervention"]["prediction_flip_rate"] for fold in folds]
        ),
    }

    for layer in ["conv1", "conv2"]:
        summary[f"{layer}_eligible_receivers"] = sum(
            graph[f"{layer}_eligible_receivers"]
            for fold in folds
            for graph in fold["graphs"]
        )
        summary[f"{layer}_single_entry_receivers"] = sum(
            graph[f"{layer}_single_entry_receivers"]
            for fold in folds
            for graph in fold["graphs"]
        )

    return summary


def print_summary(dataset_results):
    fold_columns = " | ".join(
        f"Outer fold {fold_id}"
        for fold_id in range(settings["num_folds"])
    )

    print("Attention departure from uniform weighting:")
    print(
        f"Dataset | Layer | {fold_columns} | Mean | Sample SD | "
        "Eligible receivers | Single-entry receivers"
    )

    for dataset_name in datasets:
        summary = dataset_results[dataset_name]["summary"]

        for layer, label in [("conv1", "Conv1"), ("conv2", "Conv2")]:
            departure = summary[f"{layer}_departure"]
            fold_values = " | ".join(
                f"{value:.4f}"
                for value in departure["fold_values"]
            )
            print(
                f"{dataset_name} | {label} | {fold_values} | "
                f"{departure['mean']:.4f} | {departure['sample_sd']:.4f} | "
                f"{summary[f'{layer}_eligible_receivers']} | "
                f"{summary[f'{layer}_single_entry_receivers']}"
            )

    print()
    print("Uniform-attention intervention:")
    print("Delta loss is intervention minus learned.")
    print("Delta accuracy is intervention minus learned.")
    print(f"Dataset | Measure | {fold_columns} | Mean | Sample SD")

    measures = [
        ("delta_loss", "Delta loss", 1.0, 4),
        ("delta_accuracy", "Delta accuracy (pp)", 100.0, 2),
        ("prediction_flip_rate", "Prediction flip rate (%)", 100.0, 2),
    ]

    for dataset_name in datasets:
        summary = dataset_results[dataset_name]["summary"]

        for key, label, scale, decimals in measures:
            values = summary[key]
            fold_values = " | ".join(
                f"{value * scale:.{decimals}f}"
                for value in values["fold_values"]
            )
            print(
                f"{dataset_name} | {label} | {fold_values} | "
                f"{values['mean'] * scale:.{decimals}f} | "
                f"{values['sample_sd'] * scale:.{decimals}f}"
            )


def main():
    prepare_result_path(result_path)
    source_commit = get_source_commit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"

    print()
    print("Analysis source commit:", source_commit)
    print("Device:", device)
    print("Analysis: fitted reference-GAT attention and uniform intervention")

    dataset_results = {}
    reference_source_commits = set()

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        partitions = get_partitions(dataset)

        current_settings = settings.copy()
        current_settings.update(model_settings["GAT"])
        current_settings["dataset"] = dataset_name
        current_settings["model"] = "GAT"

        reference_results = load_group(current_settings, dataset, partitions)
        reference_source_commits.add(reference_results[0]["source_commit"])

        print()
        print("Dataset:", dataset_name)
        print("Reference source commit:", reference_results[0]["source_commit"])

        folds = []

        for result in reference_results:
            fold_id = result["settings"]["fold_id"]

            print()
            print("Outer fold:", fold_id)
            print("Selected epoch:", result["selection"]["selected_epoch"])
            print("Outer-test graphs:", len(result["partitions"]["test_indices"]))
            print("Analysing fitted attention and intervention")

            folds.append(
                analyse_fold(
                    result,
                    current_settings,
                    dataset,
                    device,
                )
            )

        dataset_results[dataset_name] = {
            "folds": folds,
            "summary": summarise_dataset(folds),
        }

    if len(reference_source_commits) != 1:
        raise ValueError("Reference GAT states have different source commits")

    selected_state_count = len(datasets) * settings["num_folds"]
    reference_source_commit = next(iter(reference_source_commits))

    result = {
        "purpose": "rq4_fitted_gat_attention",
        "source_commit": source_commit,
        "reference_source_commit": reference_source_commit,
        "device": {
            "type": device.type,
            "name": device_name,
        },
        "settings": {
            "model": "GAT",
            "heads": model_settings["GAT"]["heads"],
            "num_folds": settings["num_folds"],
            "attention_measure": "one minus normalised attention entropy",
            "single_entry_receivers": "excluded from entropy and retained as counts",
            "intervention": "attention scoring tensors zeroed without retraining",
        },
        "selected_reference_states": selected_state_count,
        "new_optimisation_fits": 0,
        "datasets": dataset_results,
    }

    save_result(result_path, result)

    print()
    print("RQ4 fitted reference-GAT analysis:")
    print("Selected reference states:", selected_state_count)
    print("New optimisation fits: 0")
    print("Analysis source commit:", source_commit)
    print("Reference source commit:", reference_source_commit)

    print()
    print_summary(dataset_results)

    print()
    print("Saved result:", result_path)


if __name__ == "__main__":
    main()