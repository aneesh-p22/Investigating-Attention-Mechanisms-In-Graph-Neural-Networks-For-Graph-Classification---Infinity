import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool


class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()

        self.conv1 = GCNConv(
            num_features,
            hidden_dim,
            improved=False,
            cached=False,
            add_self_loops=True,
            normalize=True,
            bias=True,
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim,
            improved=False,
            cached=False,
            add_self_loops=True,
            normalize=True,
            bias=True,
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