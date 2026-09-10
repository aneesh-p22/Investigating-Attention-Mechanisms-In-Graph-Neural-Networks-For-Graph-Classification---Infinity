import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from src.data import load_dataset
from src.models.factory import build_model


dataset = load_dataset("MUTAG")

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)

graph_batch = next(iter(loader))

settings = {
    "model": "GAT",
    "hidden_dim": 64,
    "heads": 8,
    "uniform_attention": True,
}

model = build_model(
    settings,
    dataset.num_node_features,
    dataset.num_classes,
)

print("Model:")
print(model)

print()
print("Uniform attention:", model.uniform_attention)

print()
print("Attention scoring parameters:")

attention_parameter_names = [
    "conv1.att_src",
    "conv1.att_dst",
    "conv2.att_src",
    "conv2.att_dst",
]

parameters = dict(model.named_parameters())

for name in attention_parameter_names:
    parameter = parameters[name]

    print()
    print(name)
    print("Shape:", parameter.shape)
    print("Parameters:", parameter.numel())
    print("Trainable:", parameter.requires_grad)
    print("All zero:", bool(torch.all(parameter == 0)))

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

frozen_parameters = total_parameters - trainable_parameters

optimizer_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

print()
print("Parameter accounting:")
print("Total parameters:", total_parameters)
print("Trainable parameters:", trainable_parameters)
print("Frozen parameters:", frozen_parameters)

print(
    "Parameters supplied to optimiser:",
    sum(
        parameter.numel()
        for parameter in optimizer_parameters
    ),
)

print()
print("Trainable model parameters:")
print(
    "First message transformation:",
    model.conv1.lin.weight.requires_grad,
)
print(
    "Second message transformation:",
    model.conv2.lin.weight.requires_grad,
)
print(
    "Classifier:",
    model.classifier.weight.requires_grad,
)

print()
print("Uniform attention coefficients:")

model.eval()

with torch.no_grad():
    x = graph_batch.x

    x, (conv1_edge_index, conv1_attention) = model.conv1(
        x,
        graph_batch.edge_index,
        return_attention_weights=True,
    )

    x = torch.relu(x)

    x, (conv2_edge_index, conv2_attention) = model.conv2(
        x,
        graph_batch.edge_index,
        return_attention_weights=True,
    )

receiving_node = 0

for layer_name, attention_edge_index, attention_coefficients in [
    ("First convolution", conv1_edge_index, conv1_attention),
    ("Second convolution", conv2_edge_index, conv2_attention),
]:
    incoming_mask = (
        attention_edge_index[1] == receiving_node
    )

    incoming_edges = attention_edge_index[
        :,
        incoming_mask,
    ]

    incoming_coefficients = attention_coefficients[
        incoming_mask
    ]

    expected_coefficient = (
        1.0 / incoming_coefficients.shape[0]
    )

    print()
    print(f"{layer_name}:")
    print("Receiving node:", receiving_node)

    print("Incoming edges [source, target]:")
    print(incoming_edges.t())

    print("Incoming attention coefficients:")
    print(incoming_coefficients)

    print(
        "Expected uniform coefficient:",
        expected_coefficient,
    )

    print("Incoming coefficient sums:")
    print(incoming_coefficients.sum(dim=0))

    print(
        "Uniform:",
        bool(
            torch.allclose(
                incoming_coefficients,
                torch.full_like(
                    incoming_coefficients,
                    expected_coefficient,
                ),
            )
        ),
    )

attention_before_update = {
    name: parameters[name].detach().clone()
    for name in attention_parameter_names
}

first_message_before_update = (
    model.conv1.lin.weight.detach().clone()
)

classifier_before_update = (
    model.classifier.weight.detach().clone()
)

optimizer = Adam(
    optimizer_parameters,
    lr=0.01,
    weight_decay=0.0005,
)

model.train()

optimizer.zero_grad()

output = model(
    graph_batch.x,
    graph_batch.edge_index,
    graph_batch.batch,
)

loss = F.cross_entropy(
    output,
    graph_batch.y,
)

loss.backward()
optimizer.step()

print()
print("After one optimiser update:")
print("Loss:", loss.item())

for name in attention_parameter_names:
    parameter = parameters[name]

    print(
        f"{name} unchanged:",
        bool(
            torch.equal(
                parameter,
                attention_before_update[name],
            )
        ),
    )

    print(
        f"{name} all zero:",
        bool(torch.all(parameter == 0)),
    )

print(
    "First message transformation changed:",
    not torch.equal(
        model.conv1.lin.weight,
        first_message_before_update,
    ),
)

print(
    "Classifier changed:",
    not torch.equal(
        model.classifier.weight,
        classifier_before_update,
    ),
)