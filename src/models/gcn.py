import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Two-layer GCN that produces node embeddings for graph classification."""

    def __init__(self, in_channels, baseline_width=64):
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

    def forward(self, x, edge_index):
        # [num_nodes, in_channels] -> [num_nodes, baseline_width]
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # [num_nodes, baseline_width] -> [num_nodes, baseline_width]
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return x