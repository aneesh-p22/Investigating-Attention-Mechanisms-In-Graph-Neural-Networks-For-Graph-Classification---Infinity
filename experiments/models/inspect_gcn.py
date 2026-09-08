import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

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

toy_x = torch.ones((5, 2))
toy_batch = torch.tensor([0, 0, 0, 1, 1])

toy_output = global_add_pool(
    toy_x,
    toy_batch,
)

toy_expected = torch.tensor(
    [
        [3.0, 3.0],
        [2.0, 2.0],
    ]
)

print()
print("Toy pooled output:")
print(toy_output)
print("Toy pooling correct:", torch.equal(toy_output, toy_expected))