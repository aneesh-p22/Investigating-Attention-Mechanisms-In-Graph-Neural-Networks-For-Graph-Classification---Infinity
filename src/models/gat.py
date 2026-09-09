import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool


class GAT(nn.Module):
    """Two-layer GAT classifier using a single-head teaching configuration."""

    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()

        self.conv1 = GATConv(
            num_features,
            hidden_dim,
            heads=1,
            concat=True,
            negative_slope=0.2,
            dropout=0.0,
            add_self_loops=True,
            edge_dim=None,
            bias=True,
            residual=False,
        )

        self.conv2 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=1,
            concat=False,
            negative_slope=0.2,
            dropout=0.0,
            add_self_loops=True,
            edge_dim=None,
            bias=True,
            residual=False,
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = global_add_pool(x, batch)
        x = self.classifier(x)

        return x