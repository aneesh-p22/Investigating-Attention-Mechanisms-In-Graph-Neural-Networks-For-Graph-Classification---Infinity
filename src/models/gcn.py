import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool


class GCN(nn.Module):
    """Two-layer GCN for graph classification."""

    def __init__(self, in_channels, baseline_width, num_classes):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            baseline_width,
            improved=False,
            cached=False,
            add_self_loops=True,
            normalize=True,
            bias=True,
        )

        self.conv2 = GCNConv(
            baseline_width,
            baseline_width,
            improved=False,
            cached=False,
            add_self_loops=True,
            normalize=True,
            bias=True,
        )

        self.classifier = nn.Linear(
            baseline_width,
            num_classes,
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_add_pool(x, batch)
        x = self.classifier(x)

        return x