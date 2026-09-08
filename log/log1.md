# 1.1 Dataset Identity

- Created experiments/datasets/inspect_mutag.py to establish what MUTAG contains before using it for graph classification

    - The script loads the dataset, reports its size and available feature information, and retrieves one graph so that its PyG object type can be identified

    - This stage establishes the dataset-level structure only; the tensors stored inside individual graphs are inspected in 1.2

- Loaded MUTAG using TUDataset

    - root set to data stores the downloaded and processed dataset beneath data/MUTAG/

    - name set to MUTAG selects the MUTAG dataset

    - cleaned=False selects the standard dataset rather than the alternative cleaned version in which isomorphic duplicate graphs have been removed

    - use_node_attr=False prevents additional continuous node attributes from being appended to the node-feature matrix

        - MUTAG reports zero continuous node attributes, so there are no such values to append in this dataset

        - Its categorical node-label information is still retained and supplies the seven loaded node-feature channels

    - use_edge_attr=False prevents additional continuous edge attributes from being appended to the edge-feature matrix

        - MUTAG reports zero continuous edge attributes

        - Its categorical edge-label information is still retained, which is why four edge-feature channels remain loaded

    - transform=None means no transformation is applied whenever a processed graph is retrieved

    - pre_transform=None means no project-defined transformation is applied once during dataset processing

    - pre_filter=None means no project-defined rule removes graphs during processing

- Established the number of graph examples and prediction classes

    - Printing dataset produced MUTAG(188), while len(dataset) returned 188

        - MUTAG therefore contains 188 separate graph examples

        - Each example is one complete graph with its own nodes, edges, features and graph-level target

        - The value 188 is the number of examples available for later partitioning, not the total number of nodes across the dataset

    - type(dataset) identified the collection as a PyG TUDataset

    - dataset.num_classes returned 2

        - There are two possible graph-level target classes, making MUTAG a binary graph-classification task

        - The eventual classifier will produce two class scores for each whole graph rather than predicting a separate class for each node

        - The meaning of the numerical target values themselves has not yet been verified

- Established the node information loaded for each graph

    - dataset.num_node_features returned 7

        - Every loaded node is represented by seven numerical input values

        - A graph containing N nodes therefore has a node-feature matrix with shape [N, 7]

        - The number of nodes may differ between graphs, while the input feature width remains seven

        - This width will later determine the input dimension of the first GNN layer

    - dataset.num_node_labels returned 7

        - The processed data contains seven channels derived from categorical node labels

        - These labels describe categories of individual nodes and are model inputs, unlike the graph-level target that the model must predict

        - The individual tensor rows and the mapping between channels and real node categories have not yet been inspected

    - dataset.num_node_attributes returned 0

        - MUTAG provides no additional continuous numerical node attributes

        - The seven loaded node-feature channels therefore come from its categorical node information rather than additional continuous measurements

- Established the edge information loaded for each graph

    - dataset.num_edge_features returned 4

        - Every stored edge entry has four associated feature values

        - The value 4 is the width of the edge-feature representation, not the number of edges in a graph or the number of neighbours of a node

    - dataset.num_edge_labels returned 4

        - The processed data contains four channels derived from categorical edge labels

        - The tensor encoding and the mapping between these channels and real edge categories have not yet been inspected

    - dataset.num_edge_attributes returned 0

        - MUTAG provides no additional continuous numerical edge attributes

        - use_edge_attr=False therefore does not remove the four categorical edge-label channels

    - The principal models will deliberately ignore these edge features while retaining graph connectivity

        - Excluding edge features means their categorical information will not be supplied to the message-passing layers

        - It does not remove the edges themselves, since the GNN still requires the connectivity to determine which nodes are neighbours

- Retrieved the first graph using graph = dataset[0]

    - Indexing the dataset selects one complete graph example rather than one node

    - type(graph) identified it as a PyG Data object

        - A Data object stores the tensors describing one graph, including node features, connectivity, optional edge features and the graph-level target

    - The contents of x, edge_index, edge_attr and y, including their shapes, data types and selected values, have not yet been inspected

- Execution

    - Ran python -m experiments.datasets.inspect_mutag

    - MUTAG downloaded and processed successfully

    - Observed 188 graph examples and two graph-level classes

    - Observed seven loaded node-feature channels and no continuous node attributes

    - Observed four loaded edge-feature channels and no continuous edge attributes

    - Retrieved the first example as a PyG Data object

    - Stage 1.1 therefore established the dataset identity and the dimensions of the information loaded for nodes and edges

    - Individual tensor contents, categorical encodings, numerical target mappings, graph connectivity, graph sizes, batching and dataset partitions remain unverified

- Commit: 1.1 loaded and identified MUTAG





# 1.2 Anatomy and Connectivity

- Inspected two individual MUTAG graphs to understand the tensors stored inside each PyG Data object

    - Graph 1 contained 17 nodes and 38 stored edge entries, while Graph 2 contained 13 nodes and 28 stored edge entries

    - This confirmed that individual graphs can have different numbers of nodes and edges while retaining the same feature dimensions

- Inspected x, the node-feature matrix

    - Graph 1 had shape [17, 7] and Graph 2 had shape [13, 7]

        - Each row represents one node and the seven columns represent the seven possible categorical node types

        - The first dimension therefore changes with the number of nodes, while the feature width remains seven for every graph

    - x used the float32 dtype and its observed rows were one-hot vectors

        - A one-hot vector has one active value of 1, with its position identifying the node category

        - The floating-point representation allows the categorical values to be used as neural-network inputs; it does not make them continuous measurements

    - Verified the MUTAG node-label mapping from the dataset documentation

        - The seven channels represent C, N, O, F, I, Cl and Br respectively

        - Both inspected graphs contained C, N and O among their observed node-feature rows

        - The first five nodes in both graphs were represented by the carbon channel

- Inspected edge_index, which stores graph connectivity

    - Graph 1 had shape [2, 38] and Graph 2 had shape [2, 28]

        - Each column represents one stored source-to-destination edge entry

        - The first row contains source-node indices and the second row contains destination-node indices

    - edge_index used the int64 dtype because it stores node indices rather than feature measurements

    - Both graphs were undirected with reciprocal storage

        - An undirected connection is represented by entries in both directions, such as 0 to 1 and 1 to 0

        - Graph 1 therefore had 19 undirected edges represented by 38 stored entries

        - Graph 2 had 14 undirected edges represented by 28 stored entries

- Inspected edge_attr, which stores the categorical feature associated with each stored edge entry

    - Graph 1 had shape [38, 4] and Graph 2 had shape [28, 4]

        - Each edge_attr row corresponds to the edge_index column at the same position

        - The matching first dimensions therefore confirm one edge-feature row per stored edge entry

    - edge_attr used float32 one-hot vectors

    - Verified the four MUTAG edge-label channels as aromatic, single, double and triple bonds

        - Aromatic, single and double bonds occurred among the distinct rows in both inspected graphs

        - A triple-bond encoding was not observed in these two examples

    - These edge features are inspected so that their meaning is understood, but the principal models will deliberately exclude them while still using edge_index for connectivity

- Inspected y, the graph-level prediction target

    - y had shape [1] and dtype int64 in both graphs, meaning each complete graph has one integer class index

    - Verified that PyG maps the original MUTAG class labels to 0 for non-mutagenic and 1 for mutagenic

        - Graph 1 had y=1 and was therefore a mutagenic example

        - Graph 2 had y=0 and was therefore a non-mutagenic example

    - This graph-level target is different from the categorical node and edge labels, which describe components within the molecule rather than the class to be predicted

- Examined the processed graph connectivity

    - Both inspected graphs used valid node indices, so every stored edge referred to nodes within the corresponding graph

    - Both were reported as undirected with reciprocal edge entries

    - Neither contained self-loops in the processed representation

        - PyG removes self-loops while processing TU datasets, so model layers that require self-connections can add them later according to their own definitions

    - Neither contained duplicate stored edge entries

        - Reciprocal entries are not duplicates because the source and destination are reversed

        - PyG coalesces the TU edge representation during processing, removing repeated directed entries

    - Neither inspected graph contained isolated nodes, so every node in these examples was connected to at least one other node

- Traced the receiving neighbourhood of node 0 directly from edge_index

    - In Graph 1, node 0 had stored incoming edges from nodes 1 and 5

    - In Graph 2, node 0 had stored incoming edges from nodes 1 and 9

    - These entries show how edge_index determines which neighbouring nodes can provide information when a GNN updates node 0

    - Because the graphs use reciprocal undirected storage, these incoming nodes are also node 0's ordinary graph neighbours

- Execution

    - Ran python -m experiments.datasets.inspect_mutag

    - Successfully inspected the shapes, dtypes and selected values of x, edge_index, edge_attr and y for two MUTAG graphs

    - Verified one-hot node and edge encodings and their documented category mappings

    - Verified the graph-target mapping and observed one example from each class

    - Examined reciprocal storage, index validity, self-loops, duplicate entries, isolated nodes and a receiving neighbourhood

    - Full dataset statistics, batching and dataset partitions remain unverified and belong to later Stage 1 work

- Commit: 1.2 inspected MUTAG graph tensors and connectivity





# 1.3 Dataset Statistics

- Extended experiments/datasets/inspect_mutag.py with a short loop over all 188 graphs to establish dataset-wide class counts, graph sizes and feature dimensions

    - The loop records only the information required for the summary rather than printing or individually inspecting every graph

- Counted the graph-level target classes using graph.y.item()

    - Observed 63 graphs with class 0 and 125 graphs with class 1, accounting for all 188 examples

    - MUTAG is therefore class-imbalanced, with class 1 occurring almost twice as often as class 0

    - This matters for later dataset partitioning because small random partitions could otherwise contain distorted class proportions

    - The agreed development partition will therefore use stratification so that each partition retains the class distribution as closely as its integer size permits

- Recorded the number of nodes in every graph using graph.num_nodes

    - Observed a minimum of 10 nodes, a mean of 17.93 and a maximum of 28

    - Graph size therefore varies across MUTAG rather than every example containing the same number of nodes

    - The model must handle this variable number of nodes while accepting the same feature width for each node

    - Larger graphs also require more node representations and message-passing operations, so graph size contributes to computational cost

- Recorded graph.num_edges for every graph

    - Observed between 20 and 66 stored edge entries per graph, with a mean of 39.59

    - These values describe the number of entries stored in edge_index rather than a separately verified dataset-wide count of undirected molecular edges

    - Reciprocal undirected storage was established for the two graphs inspected in 1.2, but this statistics loop did not repeat that connectivity test across every graph

- Established node-feature consistency across the dataset

    - Added graph.x.shape[1] from every graph to a set and obtained {7}

    - A set retains each different value only once, so this establishes that all 188 graphs have seven node-feature columns

    - Graphs therefore have node-feature matrices of shape [N, 7], where N varies with graph size but the feature width remains fixed

    - This consistent width allows the same first GNN layer to process every graph and determines its input width as seven

- Established edge-feature consistency across the dataset

    - Added graph.edge_attr.shape[1] from every graph to a set and obtained {4}

    - Every MUTAG graph therefore has four edge-feature columns

    - These categorical edge features remain consistently represented even though the principal models will deliberately exclude them from their inputs

- Execution

    - Ran python -m experiments.datasets.inspect_mutag

    - Observed class counts of 63 for class 0 and 125 for class 1

    - Observed 10 to 28 nodes per graph with a mean of 17.93

    - Observed 20 to 66 stored edge entries per graph with a mean of 39.59

    - Confirmed a node-feature width of seven and an edge-feature width of four across all 188 graphs

    - Graph minibatching and development partitioning remain unverified and are handled in the next Stage 1 substages

- Commit: 1.3 recorded MUTAG dataset statistics