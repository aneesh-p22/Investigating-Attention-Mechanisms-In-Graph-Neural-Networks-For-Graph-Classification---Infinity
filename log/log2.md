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