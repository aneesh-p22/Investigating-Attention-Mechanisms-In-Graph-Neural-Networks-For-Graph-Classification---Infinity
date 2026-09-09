import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from src.data import load_dataset
from src.models.graphsage import GraphSAGE


dataset = load_dataset("MUTAG")

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)

graph_batch = next(iter(loader))

model = GraphSAGE(
    dataset.num_node_features,
    64,
    dataset.num_classes,
)

model.eval()

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

with torch.no_grad():
    x = graph_batch.x
    print("Input node features:", x.shape)

    x = model.conv1(
        x,
        graph_batch.edge_index,
    )
    print("After first convolution:", x.shape)

    x = torch.relu(x)
    print("After first ReLU:", x.shape)

    x = model.conv2(
        x,
        graph_batch.edge_index,
    )
    print("After second convolution:", x.shape)

    x = torch.relu(x)
    print("After second ReLU:", x.shape)

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
    print("Model output shape:", output.shape)