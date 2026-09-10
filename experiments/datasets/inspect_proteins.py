import torch
from torch_geometric.loader import DataLoader

from src.data import load_dataset, stratified_split


dataset = load_dataset("PROTEINS")

print(f"Dataset: {dataset}")
print(f"Dataset type: {type(dataset)}")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of graph classes: {dataset.num_classes}")

print()

print(f"Loaded node features: {dataset.num_node_features}")
print(f"Node label channels: {dataset.num_node_labels}")
print(f"Available continuous node attributes: {dataset.num_node_attributes}")

print()

print(f"Loaded edge features: {dataset.num_edge_features}")
print(f"Edge label channels: {dataset.num_edge_labels}")
print(f"Available continuous edge attributes: {dataset.num_edge_attributes}")

print()

print("Dataset meanings")
print("Graph task: enzyme versus non-enzyme protein classification")
print("Nodes: secondary structure elements")
print(
    "Edges: sequence-neighbour or spatially close "
    "secondary structure elements"
)
print(
    "Graph targets: processed numeric class IDs; "
    "no biological meaning is assigned to 0 or 1 here"
)

print()

print("Project input policy")
print("Used node information: categorical node-label channels")
print("Excluded node information: continuous node attributes")
print("Used edge information: connectivity")
print("Excluded edge information: edge features")

representative_indices = []
seen_classes = set()

for index, graph in enumerate(dataset):
    class_id = graph.y.item()

    if class_id not in seen_classes:
        representative_indices.append(index)
        seen_classes.add(class_id)

    if len(representative_indices) == dataset.num_classes:
        break

for graph_number, index in enumerate(
    representative_indices,
    start=1,
):
    graph = dataset[index]
    node_features = getattr(graph, "x", None)
    edge_features = getattr(graph, "edge_attr", None)

    print()
    print(f"Graph {graph_number}")
    print(f"Dataset index: {index}")
    print(graph)
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Stored edge entries: {graph.num_edges}")

    print()
    print("x - loaded node features")

    if node_features is None:
        print("Value: None")
    else:
        print(f"Shape: {node_features.shape}")
        print(f"Dtype: {node_features.dtype}")

    print()
    print("edge_index - connectivity")
    print(f"Shape: {graph.edge_index.shape}")
    print(f"Dtype: {graph.edge_index.dtype}")

    print()
    print("edge_attr - loaded edge features")

    if edge_features is None:
        print("Value: None")
    else:
        print(f"Shape: {edge_features.shape}")
        print(f"Dtype: {edge_features.dtype}")
        print("Distinct rows:")
        print(edge_features.unique(dim=0))

    print()
    print("y - graph target")
    print(f"Shape: {graph.y.shape}")
    print(f"Dtype: {graph.y.dtype}")
    print(f"Value: {graph.y}")

node_feature_tensors = [
    graph.x
    for graph in dataset
    if getattr(graph, "x", None) is not None
]

print()
print("Categorical node-feature rows")

if node_feature_tensors:
    all_node_features = torch.cat(
        node_feature_tensors,
        dim=0,
    )

    print(all_node_features.unique(dim=0))
else:
    print("No loaded node features")

class_counts = [0] * dataset.num_classes
node_counts = []
edge_counts = []
node_feature_widths = set()
edge_feature_widths = set()

for graph in dataset:
    class_counts[graph.y.item()] += 1
    node_counts.append(graph.num_nodes)
    edge_counts.append(graph.num_edges)

    node_features = getattr(graph, "x", None)
    edge_features = getattr(graph, "edge_attr", None)

    if node_features is None:
        node_feature_widths.add(0)
    else:
        node_feature_widths.add(node_features.shape[1])

    if edge_features is None:
        edge_feature_widths.add(0)
    else:
        edge_feature_widths.add(edge_features.shape[1])

mean_nodes = sum(node_counts) / len(node_counts)
mean_edges = sum(edge_counts) / len(edge_counts)

print()
print("Dataset statistics")

for class_id, count in enumerate(class_counts):
    print(f"Class {class_id} graphs: {count}")

print(
    f"Nodes per graph: min={min(node_counts)}, "
    f"mean={mean_nodes:.2f}, max={max(node_counts)}"
)

print(
    f"Stored edge entries per graph: min={min(edge_counts)}, "
    f"mean={mean_edges:.2f}, max={max(edge_counts)}"
)

print(f"Node feature widths: {node_feature_widths}")
print(f"Edge feature widths: {edge_feature_widths}")

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)

graph_batch = next(iter(loader))

print()
print("Graph minibatch")
print(f"Batch type: {type(graph_batch)}")
print(f"Number of graphs: {graph_batch.num_graphs}")

print()
print("Batched graph tensors")

if graph_batch.x is None:
    print("x: None")
else:
    print(f"x shape: {graph_batch.x.shape}")

print(f"edge_index shape: {graph_batch.edge_index.shape}")

if graph_batch.edge_attr is None:
    print("edge_attr: None")
else:
    print(f"edge_attr shape: {graph_batch.edge_attr.shape}")

print(f"y shape: {graph_batch.y.shape}")
print(f"Graph labels: {graph_batch.y}")

print()
print("Graph membership")
print(f"batch shape: {graph_batch.batch.shape}")
print(f"Graph IDs in batch: {graph_batch.batch.unique()}")
print(f"ptr: {graph_batch.ptr}")

split_seed = 0

train_indices, val_indices, test_indices = stratified_split(
    dataset,
    split_seed,
)

print()
print("Development split")
print(f"Split seed: {split_seed}")

for split_name, split_indices in [
    ("Training", train_indices),
    ("Validation", val_indices),
    ("Development test", test_indices),
]:
    split_class_counts = [0] * dataset.num_classes

    for index in split_indices:
        split_class_counts[dataset[index].y.item()] += 1

    print()
    print(f"{split_name} graphs: {len(split_indices)}")
    print(f"{split_name} class counts: {split_class_counts}")