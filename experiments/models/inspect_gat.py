import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from src.data import load_dataset
from src.models.gat import GAT


dataset = load_dataset("MUTAG")

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)

graph_batch = next(iter(loader))

model = GAT(
    dataset.num_node_features,
    64,
    dataset.num_classes,
    heads=8,
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

    x, (conv1_edge_index, conv1_attention) = model.conv1(
        x,
        graph_batch.edge_index,
        return_attention_weights=True,
    )
    print("After first convolution:", x.shape)

    x = torch.relu(x)
    print("After first ReLU:", x.shape)

    x, (conv2_edge_index, conv2_attention) = model.conv2(
        x,
        graph_batch.edge_index,
        return_attention_weights=True,
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

    print()
    print("Attention coefficients (untrained):")
    print("Original edge index shape:", graph_batch.edge_index.shape)

    receiving_node = 0

    for layer_name, attention_edge_index, attention_coefficients in [
        ("First convolution", conv1_edge_index, conv1_attention),
        ("Second convolution", conv2_edge_index, conv2_attention),
    ]:
        print()
        print(f"{layer_name}:")
        print("Returned edge index shape:", attention_edge_index.shape)
        print("Attention coefficient shape:", attention_coefficients.shape)
        print("Receiving node:", receiving_node)

        incoming_mask = attention_edge_index[1] == receiving_node

        incoming_edges = attention_edge_index[:, incoming_mask]
        incoming_coefficients = attention_coefficients[incoming_mask]

        print("Incoming edges [source, target]:")
        print(incoming_edges.t())

        print("Incoming attention coefficients:")
        print(incoming_coefficients)

        print("Incoming coefficient sums:")
        print(incoming_coefficients.sum(dim=0))