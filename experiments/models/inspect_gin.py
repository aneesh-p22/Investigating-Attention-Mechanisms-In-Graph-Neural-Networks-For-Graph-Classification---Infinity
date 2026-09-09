import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from src.data import load_dataset
from src.models.gin import GIN


dataset = load_dataset("MUTAG")

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)

graph_batch = next(iter(loader))

model = GIN(
    dataset.num_node_features,
    64,
    dataset.num_classes,
)

print("Model:")
print(model)

print()
print("Parameters:")

for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.numel())

print(
    "Classifier parameters:",
    sum(parameter.numel() for parameter in model.classifier.parameters()),
)

print(
    "Total parameters:",
    sum(parameter.numel() for parameter in model.parameters()),
)

print()
print("Buffers:")

for name, buffer in model.named_buffers():
    print(name, buffer.shape, buffer)

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

x = global_add_pool(
    x,
    graph_batch.batch,
)
print("After sum pooling:", x.shape)

x = model.classifier(x)
print("After classifier:", x.shape)

output = model(
    graph_batch.x,
    graph_batch.edge_index,
    graph_batch.batch,
)

print()
print("Model output:", output.shape)