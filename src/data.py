import random

import numpy as np
import torch
from torch_geometric.datasets import TUDataset


def load_dataset(name):
    return TUDataset(
        root="data",
        name=name,
        cleaned=False,
        use_node_attr=False,
        use_edge_attr=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_split(dataset, seed):
    class_indices = [
        [] for _ in range(dataset.num_classes)
    ]

    for index, graph in enumerate(dataset):
        class_indices[graph.y.item()].append(index)

    rng = random.Random(seed)

    train_indices = []
    val_indices = []
    test_indices = []

    for indices in class_indices:
        rng.shuffle(indices)

        train_end = int(0.8 * len(indices))
        val_end = train_end + int(0.1 * len(indices))

        train_indices += indices[:train_end]
        val_indices += indices[train_end:val_end]
        test_indices += indices[val_end:]

    return train_indices, val_indices, test_indices


def stratified_folds(dataset, num_folds, seed):
    class_indices = [
        [] for _ in range(dataset.num_classes)
    ]

    for index, graph in enumerate(dataset):
        class_indices[graph.y.item()].append(index)

    rng = np.random.default_rng(seed)

    outer_folds = [
        [] for _ in range(num_folds)
    ]

    fold_id = 0

    for indices in class_indices:
        shuffled_indices = np.array(indices)
        rng.shuffle(shuffled_indices)

        for index in shuffled_indices:
            outer_folds[fold_id].append(int(index))
            fold_id = (fold_id + 1) % num_folds

    for outer_fold in outer_folds:
        outer_fold.sort()

    return outer_folds


def stratified_validation_split(
    dataset,
    remainder_indices,
    seed,
):
    class_indices = [
        [] for _ in range(dataset.num_classes)
    ]

    for index in remainder_indices:
        graph = dataset[index]
        class_indices[graph.y.item()].append(index)

    rng = np.random.default_rng(seed)

    fit_indices = []
    val_indices = []

    for indices in class_indices:
        if len(indices) < 2:
            raise ValueError("Each class must contain at least two outer-remainder graphs")

        shuffled_indices = np.array(indices)
        rng.shuffle(shuffled_indices)

        val_size = int(len(indices) / 8 + 0.5)
        val_size = max(
            1,
            min(val_size, len(indices) - 1),
        )

        val_indices += [
            int(index)
            for index in shuffled_indices[:val_size]
        ]

        fit_indices += [
            int(index)
            for index in shuffled_indices[val_size:]
        ]

    fit_indices.sort()
    val_indices.sort()

    return fit_indices, val_indices
