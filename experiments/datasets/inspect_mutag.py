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

graph = dataset[0]

print()
print(f"Type of the first graph: {type(graph)}")