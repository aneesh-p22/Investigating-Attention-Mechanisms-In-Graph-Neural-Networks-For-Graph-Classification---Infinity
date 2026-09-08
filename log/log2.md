# 2.1 GCN Layers

- Added src/models/gcn.py containing the first GNN model in the project

    - GCN inherits from torch.nn.Module

        - nn.Module is PyTorch's base class for neural-network models

        - Inheriting from it allows PyTorch to keep track of the model's trainable layers and parameters

    - super().__init__() initialises the nn.Module part of the class so that layers assigned to the model are registered correctly

- Added two GCNConv message-passing layers

    - GCNConv is PyTorch Geometric's implementation of the Graph Convolutional Network operation

    - A GCN layer updates each node using its own representation and the representations of neighbouring nodes

        - Neighbour contributions are degree-normalised so that aggregation is scaled according to the connectivity of the nodes

        - A learned linear transformation then converts the aggregated information into the requested output feature width

    - The first layer is GCNConv(in_channels, 64)

        - in_channels is the number of features supplied for each input node

        - MUTAG has 7 loaded node features, so this layer receives a matrix with 7 values per node

        - The layer produces 64 learned features for every node

    - The second layer is GCNConv(64, 64)

        - It receives the 64-dimensional node representations produced by the first layer

        - It produces another 64-dimensional representation for every node

    - Two message-passing layers mean that a final node representation can incorporate information propagated across up to two graph hops

- Used the fixed GCNConv settings required by the project

    - improved=False uses the standard GCN self-loop weighting rather than PyG's alternative improved formulation

    - cached=False means the normalised graph connectivity is recomputed for each input instead of being stored and reused

        - This is appropriate because training will use minibatches containing different graphs

    - add_self_loops=True adds a self-connection for each node inside GCNConv

        - This allows a node's own current representation to contribute to its next representation as well as its neighbours' representations

    - normalize=True applies the GCN symmetric degree normalisation to the connectivity

        - This corresponds to scaling the adjacency matrix using node degrees rather than simply summing all neighbouring features without adjustment

    - bias=True gives each output feature one additional learned bias value

- Added ReLU after both GCN layers

    - torch.relu applies ReLU element by element, replacing negative values with zero while leaving positive values unchanged

    - ReLU introduces a non-linearity between graph convolution operations

    - It changes feature values but does not change the tensor shape

    - ReLU has no trainable parameters of its own

- Defined the model's forward method

    - forward describes the sequence of operations performed when the model is called

    - It receives x and edge_index

        - x contains one feature vector for every node in the minibatch

        - edge_index describes which nodes are connected and therefore which node representations can be exchanged during message passing

    - The sequence is first GCNConv, ReLU, second GCNConv, then ReLU

    - edge_attr is not passed to the model

        - MUTAG contains edge labels, but edge features are deliberately excluded from the principal models in this investigation

    - The current forward method returns node embeddings

        - It does not yet combine nodes into graph representations or make class predictions

- Added experiments/models/inspect_gcn.py to inspect the new model on a real MUTAG minibatch

    - load_dataset("MUTAG") loads the same MUTAG representation established in Stage 1

    - DataLoader groups graphs into minibatches

        - batch_size=32 requests up to 32 graphs in one minibatch

        - shuffle=False keeps the dataset order fixed for this inspection so that the inspected batch is reproducible

    - next(iter(loader)) retrieves the first minibatch produced by the DataLoader

    - The GCN is created with dataset.num_node_features as its input width

        - dataset.num_node_features is 7 for the loaded MUTAG representation

        - The fixed contextual baseline width is 64

- Inspected the model structure

    - Printing the model showed:

        - conv1 as GCNConv(7, 64)

        - conv2 as GCNConv(64, 64)

    - This confirms that the implemented dimensions match the intended two-layer architecture

- Inspected the trainable parameters with model.named_parameters()

    - named_parameters() provides each registered trainable tensor together with the name PyTorch gives it

    - parameter.shape shows the dimensions of that tensor

    - parameter.numel() gives the total number of scalar values contained in it

    - conv1.lin.weight had shape [64, 7]

        - PyTorch stores this linear weight tensor as output features by input features

        - It therefore contains 64 x 7 = 448 learned weights

    - conv1.bias had shape [64]

        - There is one learned bias for each of the 64 output features

        - The first GCN layer therefore contains 448 + 64 = 512 trainable parameters

    - conv2.lin.weight had shape [64, 64]

        - It contains 64 x 64 = 4,096 learned weights

    - conv2.bias had shape [64]

        - The second GCN layer therefore contains 4,096 + 64 = 4,160 trainable parameters

    - The complete two-layer GCN contains 512 + 4,160 = 4,672 trainable parameters

        - This matches the reported total of 4,672

- Inspected the tensor shapes through each model operation

    - The input x had shape [585, 7]

        - The first minibatch contains 32 graphs with 585 nodes in total

        - Each of those 585 nodes has the 7 loaded MUTAG node features

    - After conv1, the shape was [585, 64]

        - The number of nodes remains 585 because graph convolution updates node representations rather than combining or removing nodes

        - The feature width changes from 7 to 64 because the first GCN layer produces 64 output features per node

    - After the first ReLU, the shape remained [585, 64]

        - ReLU changes values only

    - After conv2, the shape remained [585, 64]

        - There is still one representation for every node

        - The second layer keeps the fixed 64-feature width

    - After the second ReLU, the shape again remained [585, 64]

- Manually stepped through the layers before calling the complete model

    - Accessing model.conv1 and model.conv2 directly makes the intermediate representations visible for inspection

    - Calling model(graph_batch.x, graph_batch.edge_index) then runs the forward method normally

    - The complete model output had shape [585, 64]

        - This matches the manually inspected sequence and confirms that forward performs the intended operations

- The output at this stage is still node-level rather than graph-level

    - The minibatch contains 32 graphs, but the model returns 585 node vectors because no readout operation has yet been added

    - Stage 2.2 will use graph membership information to sum the node vectors separately for each graph, producing one 64-dimensional representation per graph

- Execution

    - Ran python -m experiments.models.inspect_gcn successfully

    - The implemented model structure, parameter counts and tensor dimensions matched the intended Stage 2.1 design

- Commit: 2.1 added and inspected GCN layers





# 2.2 Sum Pooling and Classifier

- Extended the GCN from a node-embedding model into a complete graph classifier

    - Stage 2.1 ended with one 64-dimensional representation for every node in the minibatch

    - Graph classification requires one representation for each whole graph rather than one representation for each node

    - The completed forward path is now two GCNConv and ReLU blocks, global sum pooling, then a linear classifier

    - The model therefore changes representations from node features, to learned node embeddings, to graph embeddings, and finally to graph-class logits

- Imported global_add_pool from PyTorch Geometric

    - global_add_pool is a graph-level readout operation

    - A readout converts the final node representations belonging to one graph into a single representation of that graph

    - global_add_pool performs this readout by adding the node representations element by element

        - If a graph has final node representations h_1, h_2, ..., h_n, its graph representation is h_1 + h_2 + ... + h_n

    - The operation uses the batch vector to determine which nodes belong to the same graph

        - The batch vector contains one integer graph ID for every node in the minibatch

        - All nodes with graph ID 0 are summed into the representation for graph 0

        - All nodes with graph ID 1 are summed into the representation for graph 1, and similarly for the remaining graphs

    - This is different from edge_index

        - edge_index describes which nodes are connected and is used during GCN message passing

        - batch describes which whole graph each node belongs to and is used during graph-level pooling

- Sum pooling is permutation invariant

    - The numerical order assigned to the nodes of a graph is arbitrary and should not determine the graph representation

    - Addition gives the same result when its terms are reordered

    - Reordering the node representations within a graph therefore does not change their summed representation

    - This establishes invariance to node ordering

        - It does not imply that every possible pair of different graphs must always produce different summed representations

- Added nn.Linear as the graph classifier

    - nn.Linear applies a learned affine transformation to each pooled graph representation

    - The classifier receives a graph representation with width baseline_width

        - baseline_width is fixed at 64 for the GCN

    - The classifier produces num_classes output values for each graph

        - MUTAG has 2 graph classes, so the classifier maps each 64-dimensional graph representation to 2 output values

    - These two output values are logits

        - A logit is a raw score produced for one class

        - For MUTAG, each output row contains one score for class 0 and one score for class 1

        - The logits are not probabilities and do not need to sum to one

    - No softmax is applied inside the model

        - The cross-entropy loss introduced during training will operate directly on these logits

- Updated the GCN constructor to receive num_classes

    - in_channels still gives the number of loaded features for each input node

    - baseline_width still controls the 64-dimensional hidden representation width

    - num_classes now determines the number of outputs produced by the graph classifier

    - The classifier is stored as self.classifier so PyTorch registers its weights and bias as trainable model parameters

- Updated the GCN forward method to receive batch

    - x contains the feature representation for every node in the minibatch

    - edge_index contains the connectivity used by the two GCNConv layers

    - batch contains the graph membership of every node and is required by global_add_pool

    - The first GCNConv and ReLU produce the first set of learned node representations

    - The second GCNConv and ReLU produce the final node representations

    - global_add_pool then sums those node representations separately for each graph

    - self.classifier converts every pooled graph representation into its class logits

    - The model therefore now returns graph-level predictions rather than node embeddings

- Updated experiments/models/inspect_gcn.py to inspect the complete graph classifier

    - The same first MUTAG minibatch of 32 graphs is used so that the new operations can be compared directly with the Stage 2.1 inspection

    - The model is now constructed using dataset.num_classes in addition to the node-feature width and baseline width

        - dataset.num_classes is 2 for MUTAG

        - This makes the classifier output width match the two MUTAG target classes

    - The script manually applies the two GCN layers and ReLU operations before applying the new pooling and classifier operations

        - This exposes the representation shape at every meaningful boundary of the complete model

    - The complete model is then called normally using graph_batch.x, graph_batch.edge_index and graph_batch.batch

        - This checks the output produced by the actual forward method after the individual operations have been inspected

- Inspected the updated model structure

    - The model contains GCNConv(7, 64) as conv1

        - This is unchanged from Stage 2.1

    - The model contains GCNConv(64, 64) as conv2

        - This is also unchanged from Stage 2.1

    - The model now additionally contains Linear(in_features=64, out_features=2, bias=True) as classifier

        - in_features=64 matches the width of the pooled graph representation

        - out_features=2 matches the two MUTAG graph classes

        - bias=True means the classifier learns one additional bias value for each of its two outputs

- Inspected the new classifier parameters

    - classifier.weight had shape [2, 64]

        - PyTorch stores the linear weight tensor as output features by input features

        - The classifier therefore contains 2 x 64 = 128 learned weights

    - classifier.bias had shape [2]

        - There is one learned bias value for each of the two class logits

    - The classifier therefore contains 128 + 2 = 130 trainable parameters

    - The two GCN layers previously contained 4,672 trainable parameters

    - Adding the 130 classifier parameters gives 4,802 trainable parameters for the complete graph classifier

    - The reported values matched these calculations

        - Classifier parameters: 130

        - Total parameters: 4,802

    - global_add_pool adds no trainable parameters

        - It performs a fixed sum according to graph membership rather than learning weights of its own

- Inspected the representation shapes through the complete model

    - The input had shape [585, 7]

        - The minibatch contains 585 nodes across 32 MUTAG graphs

        - Each node starts with the 7 loaded MUTAG node features

    - After conv1, the shape was [585, 64]

        - There was still one row per node, but each node now had a 64-dimensional learned representation

    - After the first ReLU, the shape remained [585, 64]

        - ReLU changed values without changing the dimensions

    - After conv2, the shape remained [585, 64]

        - The second message-passing layer kept one 64-dimensional representation for each of the 585 nodes

    - After the second ReLU, the shape again remained [585, 64]

    - After global sum pooling, the shape changed to [32, 64]

        - The first dimension changed from 585 nodes to 32 graphs

        - This confirms that nodes were combined separately according to their graph membership

        - The second dimension remained 64 because summing node vectors does not change their feature width

        - Each of the 32 graphs was therefore represented by one 64-dimensional vector

    - After the classifier, the shape changed to [32, 2]

        - There was still one row for each of the 32 graphs

        - The 64 graph features were transformed into 2 logits, one for each MUTAG class

    - Calling the complete model also returned shape [32, 2]

        - This matches the manually inspected sequence

        - The forward method therefore produces one pair of class logits for every graph in the minibatch as intended

- Added a small two-graph example to demonstrate global_add_pool independently of the GCN

    - torch.ones((5, 2)) created five node representations

        - Each node had the two-dimensional representation [1, 1]

    - toy_batch was [0, 0, 0, 1, 1]

        - The first three node representations were assigned to graph 0

        - The final two node representations were assigned to graph 1

    - global_add_pool summed the node representations separately for the two graph IDs

    - Graph 0 contained three copies of [1, 1]

        - Its expected pooled representation was therefore [3, 3]

    - Graph 1 contained two copies of [1, 1]

        - Its expected pooled representation was therefore [2, 2]

    - The actual pooled output was:

        - [3, 3] for graph 0

        - [2, 2] for graph 1

    - torch.equal compared the complete actual tensor with the manually specified expected tensor

        - Toy pooling correct was True

        - This establishes on a simple known example that global_add_pool grouped nodes by their batch IDs and summed them as expected

- Execution

    - Ran python -m experiments.models.inspect_gcn successfully

    - The complete GCN contained 4,802 trainable parameters

    - Sum pooling changed the real minibatch representation from [585, 64] node embeddings to [32, 64] graph embeddings

    - The classifier changed the graph embeddings from [32, 64] to [32, 2] graph-class logits

    - The complete forward method returned shape [32, 2]

    - The independent two-graph pooling example produced the expected sums and returned Toy pooling correct: True

- Commit: 2.2 added sum pooling and graph classifier