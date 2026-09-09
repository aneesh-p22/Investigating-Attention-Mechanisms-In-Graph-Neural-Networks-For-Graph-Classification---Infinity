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
    512,
    dataset.num_classes,
    heads=8,
    embedding_dim=64,
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
    print("Input:", x.shape)

    x, (conv1_edge_index, conv1_attention) = model.conv1(
        x,
        graph_batch.edge_index,
        return_attention_weights=True,
    )
    print("After conv1:", x.shape)

    x = torch.nn.functional.elu(x)
    print("After ELU 1:", x.shape)

    x, (conv2_edge_index, conv2_attention) = model.conv2(
        x,
        graph_batch.edge_index,
        return_attention_weights=True,
    )
    print("After conv2:", x.shape)

    x = torch.nn.functional.elu(x)
    print("After ELU 2:", x.shape)

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

    print()

    print("Attention coefficients (untrained):")
    print("Original edge_index:", graph_batch.edge_index.shape)

    receiving_node = 0

    for layer_name, attention_edge_index, attention_weights in [
        ("conv1", conv1_edge_index, conv1_attention),
        ("conv2", conv2_edge_index, conv2_attention),
    ]:
        print()

        print(f"{layer_name}:")
        print("Returned edge_index:", attention_edge_index.shape)
        print("Attention weights:", attention_weights.shape)
        print("Receiving node:", receiving_node)

        incoming_mask = attention_edge_index[1] == receiving_node

        print("Incoming edges [source, target]:")
        print(attention_edge_index[:, incoming_mask].t())

        print("Incoming attention weights:")
        print(attention_weights[incoming_mask])

        incoming_sums = attention_weights.new_zeros(
            (graph_batch.num_nodes, attention_weights.size(1))
        )

        incoming_sums.index_add_(
            0,
            attention_edge_index[1],
            attention_weights,
        )

        print("Incoming coefficient sums for this node:")
        print(incoming_sums[receiving_node])

        print(
            "Incoming coefficients sum to one for every node and head:",
            torch.allclose(
                incoming_sums,
                torch.ones_like(incoming_sums),
                atol=1e-6,
                rtol=0.0,
            ),
        )