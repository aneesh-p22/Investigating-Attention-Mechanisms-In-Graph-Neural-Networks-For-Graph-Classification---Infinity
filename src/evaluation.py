import torch
import torch.nn.functional as F


def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_graphs = 0

    with torch.no_grad():
        for graph_batch in loader:
            graph_batch = graph_batch.to(device)

            logits = model(
                graph_batch.x,
                graph_batch.edge_index,
                graph_batch.batch,
            )

            loss = F.cross_entropy(
                logits,
                graph_batch.y,
            )

            predictions = logits.argmax(dim=1)
            num_graphs = graph_batch.num_graphs

            total_loss += loss.item() * num_graphs
            total_correct += (
                predictions == graph_batch.y
            ).sum().item()
            total_graphs += num_graphs

    mean_loss = total_loss / total_graphs
    accuracy = total_correct / total_graphs

    return mean_loss, accuracy


def evaluate_with_predictions(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_graphs = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for graph_batch in loader:
            graph_batch = graph_batch.to(device)

            logits = model(
                graph_batch.x,
                graph_batch.edge_index,
                graph_batch.batch,
            )

            loss = F.cross_entropy(
                logits,
                graph_batch.y,
            )

            predictions = logits.argmax(dim=1)
            num_graphs = graph_batch.num_graphs

            total_loss += loss.item() * num_graphs
            total_correct += (
                predictions == graph_batch.y
            ).sum().item()
            total_graphs += num_graphs

            all_predictions += predictions.cpu().tolist()
            all_labels += graph_batch.y.cpu().tolist()

    mean_loss = total_loss / total_graphs
    accuracy = total_correct / total_graphs

    return (
        mean_loss,
        accuracy,
        all_predictions,
        all_labels,
    )