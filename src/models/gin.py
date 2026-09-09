import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class GIN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            eps=0.0,
            train_eps=False,
        )

        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            eps=0.0,
            train_eps=False,
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