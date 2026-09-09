import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from src.models.gat import GAT


dataset = TUDataset(
    root="data",
    name="MUTAG",
    cleaned=False,
    use_node_attr=False,
    use_edge_attr=False,
)

loader = DataLoader(dataset, batch_size=32, shuffle=False)
batch = next(iter(loader))

model = GAT(
    num_features=dataset.num_node_features,
    attention_total_width=512,
    num_classes=dataset.num_classes,
    heads=8,
    embedding_dim=64,
)
model.eval()

print("PyG version:", torch_geometric.__version__)
print("Configuration: reference multi-head GAT")
print("First-layer heads:", model.conv1.heads)
print("Channels per head:", model.conv1.out_channels)
print(
    "Total first-layer width:",
    model.conv1.heads * model.conv1.out_channels,
)
print("Final node embedding width:", model.conv2.out_channels)

print("\nModel:")
print(model)

print("\nParameters:")
for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.numel())

print("\nFirst convolution parameters:", sum(
    parameter.numel() for parameter in model.conv1.parameters()
))
print("Second convolution parameters:", sum(
    parameter.numel() for parameter in model.conv2.parameters()
))
print("Classifier parameters:", sum(
    parameter.numel() for parameter in model.classifier.parameters()
))

print("Total parameters:", sum(
    parameter.numel() for parameter in model.parameters()
))
print("Trainable parameters:", sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
))

with torch.no_grad():
    print("\nGraphs in batch:", batch.num_graphs)
    print("Input shape:", batch.x.shape)

    x = model.conv1(batch.x, batch.edge_index)
    print("After first convolution:", x.shape)

    x = F.elu(x)
    print("After first ELU:", x.shape)

    x = model.conv2(x, batch.edge_index)
    print("After second convolution:", x.shape)

    x = F.elu(x)
    print("After second ELU:", x.shape)

    x = global_add_pool(x, batch.batch)
    print("After sum pooling:", x.shape)

    x = model.classifier(x)
    print("After classifier:", x.shape)

    output = model(
        batch.x,
        batch.edge_index,
        batch.batch,
    )
    print("\nComplete model output shape:", output.shape)