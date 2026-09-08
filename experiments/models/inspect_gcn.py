import torch
from torch_geometric.loader import DataLoader

from src.data import load_dataset
from src.models.gcn import GCN


dataset = load_dataset("MUTAG")

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)

graph_batch = next(iter(loader))

model = GCN(
    dataset.num_node_features,
    64,
)

print("Model:")
print(model)

print()
print("Parameters:")

for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.numel())

print(
    "First GCN layer parameters:",
    sum(parameter.numel() for parameter in model.conv1.parameters()),
)
print(
    "Total parameters:",
    sum(parameter.numel() for parameter in model.parameters()),
)

print()
print("Shape transitions:")

x = graph_batch.x
print("Input:", x.shape)

x = model.conv1(
    x,
    graph_batch.edge_index,
)
print("After conv1:", x.shape)

x = torch.relu(x)
print("After ReLU 1:", x.shape)

x = model.conv2(
    x,
    graph_batch.edge_index,
)
print("After conv2:", x.shape)

x = torch.relu(x)
print("After ReLU 2:", x.shape)

output = model(
    graph_batch.x,
    graph_batch.edge_index,
)

print()
print("Model output:", output.shape)