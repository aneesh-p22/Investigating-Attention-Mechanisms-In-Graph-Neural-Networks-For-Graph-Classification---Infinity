# 1.1 Dataset Identity

- Loaded MUTAG through PyG's TUDataset using the intended dataset configuration

    - root="data" stores the downloaded and processed dataset in the local data directory

    - name="MUTAG" selects the MUTAG dataset

    - cleaned=False uses the standard MUTAG version rather than PyG's cleaned variant

    - use_node_attr=False and use_edge_attr=False exclude additional continuous node and edge attributes

        - These settings do not remove the categorical node and edge labels provided by MUTAG

    - No transform, pre_transform or pre_filter was applied, so the graphs were inspected in the representation loaded by TUDataset

- Inspected the identity and size of the dataset

    - The dataset printed as MUTAG(188), meaning 188 graph examples were loaded

    - The dataset object was a torch_geometric.datasets.tu_dataset.TUDataset

    - dataset.num_classes returned 2

        - MUTAG is therefore a binary graph-classification dataset

        - Each complete graph is one example assigned to one of two graph-level classes

- Inspected the loaded node information

    - dataset.num_node_features returned 7

        - Each node is represented by seven loaded feature values

        - A graph containing N nodes therefore has an x matrix with N rows and seven columns

    - dataset.num_node_labels returned 7

        - The seven node-feature channels come from categorical node labels

    - dataset.num_node_attributes returned 0

        - No additional continuous node attributes are loaded

- Inspected the loaded edge information

    - dataset.num_edge_features returned 4

        - Each stored edge entry has four loaded feature values

    - dataset.num_edge_labels returned 4

        - The four edge-feature channels come from categorical edge labels

    - dataset.num_edge_attributes returned 0

        - No additional continuous edge attributes are loaded

    - The edge features remain available in the dataset, although the principal GNN models will later use edge_index without passing edge_attr into their message-passing layers

- Inspected an individual dataset item

    - dataset[0] was a torch_geometric.data.data.Data object

    - PyG therefore represents each MUTAG graph as a Data object containing its graph tensors

    - The contents of those tensors were inspected in 1.2

- Execution

    - Ran python -m experiments.datasets.inspect_mutag

    - Confirmed that 188 MUTAG graphs and two graph classes were loaded

    - Confirmed seven categorical node-feature channels and four categorical edge-feature channels

    - Confirmed that no additional continuous node or edge attributes were loaded

    - Confirmed that individual graphs are represented as PyG Data objects

- Commit: 1.1 loaded and identified MUTAG


# 1.2 Anatomy and Connectivity

- Inspected two individual MUTAG graphs to understand the tensors stored inside a PyG Data object

    - Graph 1 contained 17 nodes and 38 stored edge entries

    - Graph 2 contained 13 nodes and 28 stored edge entries

    - The different graph sizes show that the number of nodes and connections can vary between molecules

- Recorded the verified meanings of the categorical features

    - The seven node-feature columns represent C, N, O, F, I, Cl and Br

        - These correspond to carbon, nitrogen, oxygen, fluorine, iodine, chlorine and bromine

    - The four edge-feature columns represent aromatic, single, double and triple bonds

    - The processed graph targets use 0 for non-mutagenic and 1 for mutagenic

    - Recording these mappings makes the one-hot feature tensors directly interpretable rather than treating the columns as unnamed categories

- Inspected x, the node-feature matrix

    - Graph 1 had shape [17, 7] and Graph 2 had shape [13, 7]

        - Each row represents one node

        - Each column represents one of the seven categorical atom types

        - The number of rows changes with graph size while the feature width remains seven

    - x used the float32 dtype

        - The values are stored as floating-point inputs for neural-network operations but represent categorical atom types rather than continuous measurements

    - The distinct x rows observed in both graphs were one-hot encodings

        - A one-hot encoding contains one value of 1 and zeros in the remaining positions

        - The position containing 1 identifies the atom type using the recorded column ordering

    - Carbon, nitrogen and oxygen occurred among the distinct node-feature rows in both inspected graphs

- Inspected edge_index, which stores graph connectivity

    - Graph 1 had shape [2, 38] and Graph 2 had shape [2, 28]

        - Each column represents one stored source-to-destination edge entry

        - The first row contains source-node indices

        - The second row contains destination-node indices

    - edge_index used the int64 dtype because its values are node indices

- Inspected edge_attr, which stores the categorical feature associated with each stored edge entry

    - Graph 1 had shape [38, 4] and Graph 2 had shape [28, 4]

        - edge_attr therefore contains one row for every stored edge_index column

        - Each row has four columns representing the possible bond categories

    - edge_attr used the float32 dtype

    - Its distinct rows were one-hot encodings using the aromatic, single, double and triple column ordering

    - Aromatic, single and double bond encodings occurred in both inspected graphs

    - A triple-bond encoding was not observed in these two examples

- Inspected y, the graph-level target

    - y had shape [1] and dtype int64 for each graph

        - The one value is a target for the complete graph rather than for an individual node or edge

    - Graph 1 had target 1 and was therefore a mutagenic example

    - Graph 2 had target 0 and was therefore a non-mutagenic example

    - The Data summary reports tensor shapes rather than tensor contents

        - For example, y=[1] in the Data summary means that y has one element and does not mean that its target value is necessarily 1

- Examined the connectivity representation of both graphs

    - Checked that every value in edge_index was at least 0 and smaller than graph.num_nodes

        - Valid node indices returned True for both inspected graphs

        - Every stored edge therefore referred to nodes that exist within its graph

    - graph.is_undirected() returned True for both graphs

        - The undirected connections are stored reciprocally, with one source-to-destination entry in each direction

    - graph.has_self_loops() returned False for both graphs

        - Neither inspected graph contained a stored edge from a node to itself

    - Used torch.unique over the columns of edge_index to check for repeated stored directed entries

        - Neither inspected graph contained duplicate stored edge entries

        - Reciprocal entries are not duplicates because their source and destination positions are reversed

    - graph.has_isolated_nodes() returned False for both graphs

        - Every node in these two examples was connected to at least one other node

    - Because the inspected graphs used reciprocal storage and contained no self-loops or duplicate entries, Graph 1's 38 stored entries represented 19 undirected connections and Graph 2's 28 stored entries represented 14

        - This conclusion applies to the two inspected examples rather than assuming that the same properties have been exhaustively checked for every graph

- Traced one receiving neighbourhood using node 0

    - Compared the destination row of edge_index with node 0 to create a Boolean incoming_mask

        - True positions identify stored edges whose destination is node 0

    - Applying the mask to edge_index selected the complete stored edge entries into node 0

    - Graph 1 had incoming entries from nodes 1 and 5

    - Graph 2 had incoming entries from nodes 1 and 9

    - This provides a concrete example of how edge_index identifies neighbouring source nodes that can provide information to a destination node during message passing

- Execution

    - Ran python -m experiments.datasets.inspect_mutag

    - Inspected the shapes, dtypes and categorical encodings of x, edge_index, edge_attr and y for two graphs

    - Recorded the verified atom, bond and graph-target mappings

    - Checked node-index bounds, reciprocal storage, self-loops, duplicate entries and isolated nodes

    - Traced one receiving neighbourhood directly from edge_index

- Commit: 1.2 inspected MUTAG graph tensors and connectivity


# 1.3 Dataset Statistics

- Extended the inspection from two representative graphs to all 188 MUTAG graphs

    - Iterated through the complete dataset and collected graph-class counts, node counts, stored edge-entry counts and feature widths

- Counted the graph-level classes

    - Class 0 contained 63 graphs

    - Class 1 contained 125 graphs

    - The counts sum to the full 188 graph examples

    - Class 1 is therefore substantially more common than Class 0

    - This makes preserving the class distribution important when the development split is created, but does not by itself justify changing the loss function or rebalancing the dataset

- Measured the number of nodes per graph

    - Minimum: 10 nodes

    - Mean: 17.93 nodes

    - Maximum: 28 nodes

    - MUTAG therefore contains relatively small graphs, but their number of nodes is not fixed

    - A GNN must consequently process graphs with different numbers of node rows while using the same feature width

- Measured the number of stored edge entries per graph

    - Minimum: 20 stored edge entries

    - Mean: 39.59 stored edge entries

    - Maximum: 66 stored edge entries

    - The amount of stored connectivity therefore also varies between graphs

    - These values are kept as stored edge-entry counts rather than automatically dividing them by two

        - Reciprocal storage, self-loops and duplicates were inspected for the two representative graphs in 1.2 but were not exhaustively checked across all 188 graphs

- Checked the node-feature width across the complete dataset

    - node_feature_widths was a Python set

        - A set stores each distinct value only once

        - Adding graph.x.shape[1] for every graph therefore provides a compact way to see whether more than one node-feature width occurs

    - The resulting set was {7}

        - Every MUTAG graph therefore had seven columns in x

        - The later GNNs can use one fixed input feature dimension even though their numbers of nodes vary

- Checked the edge-feature width across the complete dataset

    - The resulting edge_feature_widths set was {4}

        - Every graph had four columns in edge_attr

        - The edge-feature width is therefore consistent across the complete dataset

- Execution

    - Ran python -m experiments.datasets.inspect_mutag

    - Confirmed class counts of 63 and 125

    - Observed graph sizes from 10 to 28 nodes with a mean of 17.93

    - Observed 20 to 66 stored edge entries per graph with a mean of 39.59

    - Confirmed consistent node and edge feature widths of seven and four across all 188 graphs

- Commit: 1.3 recorded MUTAG dataset statistics


# 1.4 Graph Minibatching

- Created a PyG DataLoader with batch_size=32 to inspect how multiple graph examples are represented together

    - batch_size=32 combines 32 complete graph examples into one minibatch

    - shuffle=False was used for this inspection so that the batch remained in dataset order

        - The purpose here was to understand the batch representation rather than reproduce the shuffled ordering that will later be used during training

- Retrieved the first graph minibatch

    - The batch contained 32 graphs

    - Across those graphs there were 585 nodes and 1,304 stored edge entries

- Inspected the combined graph tensors

    - x had shape [585, 7]

        - Node-feature rows from the 32 graphs were concatenated into one larger matrix

        - The node-feature width remained seven

    - edge_index had shape [2, 1304]

        - The stored connectivity from all 32 graphs was represented in one edge-index tensor

    - edge_attr had shape [1304, 4]

        - There remained one four-channel edge-feature row for every stored edge entry

    - y had shape [32]

        - The individual graph targets were combined into one tensor containing one target for each of the 32 graphs

        - A later graph classifier must therefore produce one graph-level prediction for each graph in the minibatch

- Inspected graph_batch.batch, which records graph membership for individual nodes

    - graph_batch.batch had shape [585]

        - There is one graph identifier for each of the 585 nodes in the combined x matrix

        - Nodes belonging to the same original graph receive the same graph identifier

    - This membership information will later allow graph-level pooling to determine which node representations belong to each graph

- Inspected graph_batch.ptr, which stores the node boundaries between graphs

    - ptr contained 33 values for 32 graphs

        - The extra value is required because the sequence records both graph starts and the final ending position

    - The first values were 0, 17 and 30

        - The first graph occupied node positions 0 through 16 in the combined batch

        - The second graph occupied positions 17 through 29

        - The third graph began at position 30

    - The final ptr value was 585, matching the total number of nodes in the batch

    - ptr also makes the relationship between local graph indices and combined batch positions understandable

        - Each original graph numbers its own nodes starting locally from 0

        - When the graphs are combined, later graphs occupy positions after all nodes belonging to earlier graphs

        - For example, the second graph begins at position 17 because the first graph contains 17 nodes

- Verified that batching did not connect separate graphs

    - Used graph_batch.batch to find the graph identifier associated with every source node in edge_index

        - These values were stored as source_graph_ids

    - Did the same for every destination node and stored the values as destination_graph_ids

    - torch.equal returned True when the source and destination graph identifiers were compared

        - Every stored edge therefore started and ended within the same original graph

        - The minibatch behaves as a disconnected collection of graphs rather than introducing connections between different molecules

- Related the batch representation to later GNN processing

    - Message-passing layers can process the combined node and edge tensors while information remains confined to each original graph because there are no cross-graph edges

    - graph_batch.batch later tells global graph pooling which node representations should be combined together

    - The node representations can therefore be reduced back into 32 separate graph representations before graph classification

- Execution

    - Ran python -m experiments.datasets.inspect_mutag

    - Inspected a 32-graph minibatch containing 585 nodes and 1,304 stored edge entries

    - Confirmed one graph target for each graph and one graph-membership value for each node

    - Used ptr to understand the boundaries between graphs in the combined node representation

    - Verified that every stored edge remained within its original graph

- Commit: 1.4 inspected MUTAG graph minibatching





# 1.5 Development Split and Seeding

- Moved the common TU dataset loading configuration into load_dataset in src/data.py

    - load_dataset takes the dataset name and applies the same TUDataset settings previously written directly in inspect_mutag.py

    - inspect_mutag.py now loads MUTAG with load_dataset("MUTAG")

    - Re-running the full inspection produced the same MUTAG identity, graph representations, dataset statistics and minibatch results as before, confirming that the loading behaviour had not changed

- Added set_seed for controlling randomness during later experiments

    - random.seed sets Python's random state

    - np.random.seed sets NumPy's random state

    - torch.manual_seed sets PyTorch's random state

    - These will later allow model runs using the same training seed to begin from controlled random states

    - The development split uses its own split seed separately, so constructing the partition does not depend on calling set_seed first

- Added stratified_split to create the fixed development train, validation and test partitions

    - The function first creates one list of graph indices for each graph class

    - enumerate(dataset) provides each graph together with its original dataset index

    - graph.y.item() identifies the graph's class, allowing its original index to be added to the corresponding class list

    - Splitting the classes separately makes the partition stratified rather than randomly splitting all graphs together without regard to their targets

- Used random.Random(seed) to create a local random generator for the partition

    - rng.shuffle randomly changes the ordering of the indices within each class before they are divided

    - Using split seed 0 makes the selected development graph membership reproducible

    - The original graph indices are retained, so the split records exactly which MUTAG examples belong to each partition

- Divided each class approximately 80/10/10 using integer boundaries

    - train_end takes the first approximately 80% of each shuffled class for training

    - val_end then gives approximately the next 10% to validation

    - The remaining graphs are assigned to test

    - Integer graph counts mean the realised percentages do not have to be exactly 80%, 10% and 10%

- Inspected the realised MUTAG development split

    - Training contained 150 graphs

        - 50 were Class 0 and 100 were Class 1

    - Validation contained 18 graphs

        - 6 were Class 0 and 12 were Class 1

    - Test contained 20 graphs

        - 7 were Class 0 and 13 were Class 1

    - The complete dataset contains 63 Class 0 and 125 Class 1 graphs, so the three partitions retained a similar class distribution

    - The 150/18/20 sizes correspond to approximately 79.8% training, 9.6% validation and 10.6% test

        - The small difference from exact 80/10/10 is caused by converting the class-wise proportions to whole numbers of graphs

        - No adjustment was made simply to force validation and test to have identical sizes

- Printed the original graph indices belonging to each partition

    - The lists identify which examples from the original MUTAG dataset were selected for training, validation and test

    - These exact index lists can be reproduced from the same dataset using split seed 0

    - The full index lists were inspected in the program output but do not need to be duplicated in the log

- Established the role of each development partition

    - Training graphs will later be used to update model parameters

    - Validation graphs will be kept separate from parameter updates and used for model-state selection during development

    - Test graphs will remain separate from both training and validation and will be evaluated after the validation-selected state has been chosen

    - This development test partition provides preliminary engineering evidence rather than the final cross-validation results used for the main experimental conclusions

- Execution

    - Ran python -m experiments.datasets.inspect_mutag

    - Confirmed a reproducible stratified development split using seed 0

    - Observed 150 training, 18 validation and 20 test graphs

    - Observed class allocations of [50, 100], [6, 12] and [7, 13] respectively

    - Inspected the original MUTAG graph indices assigned to all three partitions

- Commit: 1.5 added stratified development split and seeding





# 1 Closing Notes

- Decisions

    - MUTAG is loaded through PyG's TUDataset using the standard dataset version with cleaned=False

        - Continuous node and edge attributes are not included

        - No transform, pre_transform or pre_filter is applied

        - This input policy is fixed for the principal MUTAG experiments

    - Categorical node features are retained as the seven loaded MUTAG node-label channels

        - The columns represent C, N, O, F, I, Cl and Br

        - These features form the node input x used by the GNNs

    - Edge connectivity is retained through edge_index

        - The four categorical edge-feature channels represent aromatic, single, double and triple bonds

        - edge_attr is inspected and understood but will not be passed into the principal model message-passing layers

        - This is a deliberate project control rather than a limitation of PyG

    - Graph targets use the processed binary labels 0 for non-mutagenic and 1 for mutagenic

    - Graph minibatches use PyG's ordinary disconnected batching representation

        - Nodes and stored edge entries from several graphs are represented together

        - batch records which graph each node belongs to

        - ptr records the node boundaries between graphs

        - No special preprocessing is required to connect graphs because separate graphs remain disconnected inside the minibatch

    - Development minibatches use batch size 32

        - shuffle=False was used only for the Stage 1 inspection so that batching could be read in dataset order

        - Training behaviour is a separate concern and is not fixed by this inspection setting

    - The development partition is stratified by graph class and uses split seed 0

        - The simple class-wise 80/10/10 procedure is retained rather than adding extra splitting infrastructure

        - Integer rounding gives a realised MUTAG split of 150 training, 18 validation and 20 test graphs

        - The unequal validation and test sizes are accepted as the natural result of the simple class-wise integer split

    - Original dataset indices are retained in the development partition

        - The split therefore identifies the exact original MUTAG graphs assigned to training, validation and test

    - Randomness used to choose the development partition is conceptually separate from the random seed used during later model training

        - The development graph membership remains fixed while different models are developed on the same partition

    - Stage 1 inspection code is intentionally straightforward rather than exhaustive

        - Checks and printed values were kept only where they helped understand a scientifically relevant property of the dataset or batching representation

        - Additional sanity checks will not be added routinely unless they are required, appropriate or important enough to remain part of the project

- Ideas

    - Edge features could support a later edge-aware extension if additional work is justified after the core investigation

        - This is not part of the principal model comparison and does not change the current edge-feature exclusion policy

    - The dataset-loading and stratified-splitting functions are written so that datasets with more than two consecutively numbered graph classes can be handled without hard-coding two class lists

        - No additional multi-class dataset is currently part of the core investigation

    - No other optional dataset-processing work is activated from Stage 1

        - The next priority is to understand and implement the first complete graph classifier rather than extending the input pipeline

- Report notes

    - MUTAG contains 188 graph examples and two graph-level classes

        - Class 0 contains 63 graphs

        - Class 1 contains 125 graphs

        - The class distribution is therefore imbalanced, with Class 1 substantially more common than Class 0

    - Each node has seven categorical input channels

        - The channels represent C, N, O, F, I, Cl and Br

        - No additional continuous node attributes are loaded

    - Each stored edge entry has four available categorical feature channels

        - The channels represent aromatic, single, double and triple bonds

        - No additional continuous edge attributes are loaded

        - These edge features are excluded from the principal model inputs while connectivity itself is retained

    - MUTAG graphs vary in size

        - Minimum nodes per graph: 10

        - Mean nodes per graph: 17.93

        - Maximum nodes per graph: 28

    - Stored connectivity also varies by graph

        - Minimum stored edge entries per graph: 20

        - Mean stored edge entries per graph: 39.59

        - Maximum stored edge entries per graph: 66

        - These values should be described as stored edge entries unless reciprocal storage has been established for the specific quantity being converted to undirected edges

    - Feature dimensions were consistent across all 188 graphs

        - Every x matrix had seven columns

        - Every edge_attr matrix had four columns

        - The fixed seven-column node input determines the input width required by the first GNN layer on MUTAG

    - Two representative graphs were inspected directly

        - Graph 1 contained 17 nodes and 38 stored edge entries with graph target 1

        - Graph 2 contained 13 nodes and 28 stored edge entries with graph target 0

        - Both inspected examples had valid node indices, reciprocal undirected storage, no self-loops, no duplicate stored directed entries and no isolated nodes

        - These connectivity observations apply directly to the inspected examples and were not treated as an exhaustive dataset-wide connectivity audit

    - The first inspected minibatch contained 32 graphs

        - The batch contained 585 nodes and 1,304 stored edge entries

        - x had shape [585, 7]

        - edge_index had shape [2, 1304]

        - edge_attr had shape [1304, 4]

        - y had shape [32], giving one graph target per graph

        - batch had one graph-membership value for each of the 585 nodes

        - ptr had 33 boundaries for the 32 graphs and ended at 585

        - Every inspected stored edge remained within one original graph in the batch

    - The development split uses seed 0 and preserves the class structure approximately within each partition

        - Training: 150 graphs with class counts [50, 100]

        - Validation: 18 graphs with class counts [6, 12]

        - Test: 20 graphs with class counts [7, 13]

        - These correspond to approximately 79.8%, 9.6% and 10.6% of the complete dataset

        - The small departure from exact 80/10/10 results from assigning whole graphs after class-wise integer rounding

    - The development partitions have different methodological roles

        - Training graphs are used for parameter optimisation

        - Validation graphs will be used for model-state selection

        - Development test graphs are evaluated after validation-based selection

        - Development results on MUTAG are preliminary engineering evidence because MUTAG is later reused in the locked final cross-validation assessment

    - Evidence available for later Methods and dataset reporting includes the actual loading policy, categorical feature mappings, graph counts, class counts, graph-size statistics, stored edge-entry statistics, feature widths, batching behaviour and fixed development-partition policy

    - Stage 1 notes are factual evidence for later report writing rather than drafted manuscript prose