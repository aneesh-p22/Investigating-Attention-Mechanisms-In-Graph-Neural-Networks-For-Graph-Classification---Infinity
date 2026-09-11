import torch
from torch_geometric.loader import DataLoader

from src.data import load_dataset, stratified_split


dataset = load_dataset("MUTAG")

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

print("Feature meanings")
print("Node feature columns: C, N, O, F, I, Cl, Br")
print("Edge feature columns: aromatic, single, double, triple")
print("Graph targets: 0 = non-mutagenic, 1 = mutagenic")

graph_1 = dataset[0]
graph_2 = dataset[1]

print()

print(f"Type of the first graph: {type(graph_1)}")

for graph_number, graph in [(1, graph_1), (2, graph_2)]:
    print()
    print(f"Graph {graph_number}")
    print(graph)
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Stored edge entries: {graph.num_edges}")

    print()
    print("x - node features")
    print(f"Shape: {graph.x.shape}")
    print(f"Dtype: {graph.x.dtype}")
    print("Distinct rows:")
    print(graph.x.unique(dim=0))

    print()
    print("edge_index - connectivity")
    print(f"Shape: {graph.edge_index.shape}")
    print(f"Dtype: {graph.edge_index.dtype}")

    print()
    print("edge_attr - edge features")
    print(f"Shape: {graph.edge_attr.shape}")
    print(f"Dtype: {graph.edge_attr.dtype}")
    print("Distinct rows:")
    print(graph.edge_attr.unique(dim=0))

    print()
    print("y - graph target")
    print(f"Shape: {graph.y.shape}")
    print(f"Dtype: {graph.y.dtype}")
    print(f"Value: {graph.y}")

    edge_index = graph.edge_index
    unique_edges = torch.unique(edge_index, dim=1)

    node_indices_valid = (
        edge_index.min().item() >= 0
        and edge_index.max().item() < graph.num_nodes
    )
    has_duplicate_entries = unique_edges.shape[1] != edge_index.shape[1]

    print()
    print("Connectivity")
    print(f"Valid node indices: {node_indices_valid}")
    print(f"Undirected with reciprocal edge entries: {graph.is_undirected()}")
    print(f"Contains self-loops: {graph.has_self_loops()}")
    print(f"Contains duplicate stored edge entries: {has_duplicate_entries}")
    print(f"Contains isolated nodes: {graph.has_isolated_nodes()}")

    node = 0
    incoming_mask = edge_index[1] == node
    edges_into_node = edge_index[:, incoming_mask]

    print(f"Stored edge entries into node {node}:")
    print(edges_into_node)

class_counts = [0] * dataset.num_classes
node_counts = []
edge_counts = []
node_feature_widths = set()
edge_feature_widths = set()

for graph in dataset:
    class_counts[graph.y.item()] += 1
    node_counts.append(graph.num_nodes)
    edge_counts.append(graph.num_edges)
    node_feature_widths.add(graph.x.shape[1])
    edge_feature_widths.add(graph.edge_attr.shape[1])

mean_nodes = sum(node_counts) / len(node_counts)
mean_edges = sum(edge_counts) / len(edge_counts)

print()
print("Dataset statistics")
print(f"Class 0 graphs: {class_counts[0]}")
print(f"Class 1 graphs: {class_counts[1]}")
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
print(f"x shape: {graph_batch.x.shape}")
print(f"edge_index shape: {graph_batch.edge_index.shape}")
print(f"edge_attr shape: {graph_batch.edge_attr.shape}")
print(f"y shape: {graph_batch.y.shape}")
print(f"Graph labels: {graph_batch.y}")

print()
print("Graph membership")
print(f"batch shape: {graph_batch.batch.shape}")
print(f"Graph IDs in batch: {graph_batch.batch.unique()}")
print(f"ptr: {graph_batch.ptr}")

source_graph_ids = graph_batch.batch[graph_batch.edge_index[0]]
destination_graph_ids = graph_batch.batch[graph_batch.edge_index[1]]

print()
print(
    "Every stored edge stays within one graph: "
    f"{torch.equal(source_graph_ids, destination_graph_ids)}"
)


split_seed = 0

train_indices, val_indices, test_indices = stratified_split(
    dataset,
    split_seed,
)

print()
print("Development split")
print(f"Split seed: {split_seed}")

for split_name, split_indices in [
    ("Train", train_indices),
    ("Validation", val_indices),
    ("Test", test_indices),
]:
    split_class_counts = [0] * dataset.num_classes

    for index in split_indices:
        split_class_counts[dataset[index].y.item()] += 1

    print()
    print(f"{split_name} graphs: {len(split_indices)}")
    print(f"{split_name} class counts: {split_class_counts}")
    print(f"{split_name} indices: {split_indices}")