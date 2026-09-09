import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_add_pool


class GraphSAGE(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()

        self.conv1 = SAGEConv(
            num_features,
            hidden_dim,
            aggr="mean",
            normalize=False,
            root_weight=True,
            project=False,
            bias=True,
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim,
            aggr="mean",
            normalize=False,
            root_weight=True,
            project=False,
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