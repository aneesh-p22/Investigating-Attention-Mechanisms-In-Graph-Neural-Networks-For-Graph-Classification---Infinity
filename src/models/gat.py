import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool


class GAT(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, heads):
        super().__init__()

        if heads < 1 or hidden_dim % heads != 0:
            raise ValueError(
                "heads must be positive and divide hidden_dim exactly"
            )

        channels_per_head = hidden_dim // heads

        self.conv1 = GATConv(
            num_features,
            channels_per_head,
            heads=heads,
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

        self.classifier = nn.Linear(
            hidden_dim,
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