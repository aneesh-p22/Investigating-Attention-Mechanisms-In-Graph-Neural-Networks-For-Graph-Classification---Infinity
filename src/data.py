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