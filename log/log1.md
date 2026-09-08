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