import torch
from torch_geometric.datasets import TUDataset


dataset = TUDataset(
    root="data",
    name="MUTAG",
    cleaned=False,
    use_node_attr=False,
    use_edge_attr=False,
    transform=None,
    pre_transform=None,
    pre_filter=None,
)

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
    print("First five rows:")
    print(graph.x[:5])
    print("Distinct rows:")
    print(graph.x.unique(dim=0))

    print()
    print("edge_index - connectivity")
    print(f"Shape: {graph.edge_index.shape}")
    print(f"Dtype: {graph.edge_index.dtype}")
    print("First ten stored edge entries:")
    print(graph.edge_index[:, :10])

    print()
    print("edge_attr - edge features")
    print(f"Shape: {graph.edge_attr.shape}")
    print(f"Dtype: {graph.edge_attr.dtype}")
    print("First ten rows:")
    print(graph.edge_attr[:10])
    print("Distinct rows:")
    print(graph.edge_attr.unique(dim=0))

    print()
    print("y - graph target")
    print(f"Shape: {graph.y.shape}")
    print(f"Dtype: {graph.y.dtype}")
    print(f"Value: {graph.y}")

    edge_index = graph.edge_index
    unique_edges = torch.unique(edge_index, dim=1)

    valid_indices = (
        edge_index.min().item() >= 0
        and edge_index.max().item() < graph.num_nodes
    )
    duplicate_entries = unique_edges.shape[1] != edge_index.shape[1]

    print()
    print("Connectivity")
    print(f"Valid node indices: {valid_indices}")
    print(f"Undirected with reciprocal edge entries: {graph.is_undirected()}")
    print(f"Contains self-loops: {graph.has_self_loops()}")
    print(f"Contains duplicate stored edge entries: {duplicate_entries}")
    print(f"Contains isolated nodes: {graph.has_isolated_nodes()}")

    node = 0
    incoming_mask = edge_index[1] == node
    neighbours = edge_index[0, incoming_mask]

    print(f"Nodes with stored edges into node {node}: {neighbours}")

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