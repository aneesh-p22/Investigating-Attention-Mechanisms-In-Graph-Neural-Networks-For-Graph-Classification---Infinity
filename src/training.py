import math
import time

import torch
import torch.nn.functional as F

from src.evaluation import evaluate


def train_epoch(model, loader, optimizer, device):
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


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
):
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_state = None
    training_seconds = 0.0

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        training_seconds += time.perf_counter() - start_time

        if not math.isfinite(train_loss):
            raise ValueError(f"Non-finite training loss at epoch {epoch}")

        val_loss, val_accuracy, _, _ = evaluate(
            model,
            val_loader,
            device,
        )

        if not math.isfinite(val_loss):
            raise ValueError(f"Non-finite validation loss at epoch {epoch}")

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy

            best_state = {
                name: value.clone()
                for name, value in model.state_dict().items()
            }

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"train loss {train_loss:.4f} | "
                f"train accuracy {train_accuracy:.4f} | "
                f"val loss {val_loss:.4f} | "
                f"val accuracy {val_accuracy:.4f}"
            )

    model.load_state_dict(best_state)

    return (
        best_epoch,
        best_val_loss,
        best_val_accuracy,
        training_seconds,
    )