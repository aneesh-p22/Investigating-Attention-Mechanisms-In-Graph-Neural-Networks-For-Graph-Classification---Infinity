import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from src.data import load_dataset, set_seed, stratified_split
from src.models.gcn import GCN


settings = {
    "dataset": "MUTAG",
    "baseline_width": 64,
    "learning_rate": 0.01,
    "weight_decay": 0.0005,
    "epochs": 1000,
    "batch_size": 32,
    "seed": 0,
    "split_seed": 0,
}


def train_epoch(model, loader, optimizer, device):
    """Train the model for one epoch and return loss and accuracy."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_graphs = 0

    for graph_batch in loader:
        graph_batch = graph_batch.to(device)

        optimizer.zero_grad()

        logits = model(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.batch,
        )

        loss = F.cross_entropy(
            logits,
            graph_batch.y,
        )

        loss.backward()
        optimizer.step()

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


def evaluate(model, loader, device):
    """Evaluate the model without updating its parameters."""
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


def main():
    dataset = load_dataset(settings["dataset"])

    train_indices, val_indices, _ = stratified_split(
        dataset,
        settings["split_seed"],
    )

    set_seed(settings["seed"])

    train_generator = torch.Generator()
    train_generator.manual_seed(settings["seed"])

    train_loader = DataLoader(
        dataset[train_indices],
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=train_generator,
    )

    val_loader = DataLoader(
        dataset[val_indices],
        batch_size=settings["batch_size"],
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = GCN(
        dataset.num_node_features,
        settings["baseline_width"],
        dataset.num_classes,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    print("Settings:")

    for name, value in settings.items():
        print(f"{name}: {value}")

    print()
    print(f"Device: {device}")
    print(f"Training graphs: {len(train_indices)}")
    print(f"Validation graphs: {len(val_indices)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    print()
    print("Training:")

    for epoch in range(1, settings["epochs"] + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device,
        )

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"train loss {train_loss:.4f} | "
                f"train accuracy {train_accuracy:.4f} | "
                f"val loss {val_loss:.4f} | "
                f"val accuracy {val_accuracy:.4f}"
            )


if __name__ == "__main__":
    main()