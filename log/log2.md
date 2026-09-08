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





# 2.3 Ordinary Fitting and Validation

- Added experiments/train.py to introduce ordinary supervised fitting and validation

    - The script trains the complete graph classifier developed in 2.1 and 2.2

    - Training now adjusts the model's parameters using labelled training graphs rather than only inspecting an untrained model

    - Validation measures the fitted model on graphs that do not directly produce parameter updates

    - The development test partition is deliberately not used in this substage

        - Test evaluation is deferred until validation-based model selection and state restoration are implemented in 2.4

- Kept the training implementation simple while preparing for the other GNN models that are already planned

    - train_epoch receives a model as an argument rather than depending on GCN-specific layers

    - evaluate also receives a model rather than referring directly to GCN internals

    - Both functions only assume that the graph classifier receives x, edge_index and batch and returns graph-level logits

        - This is the interface that the later GraphSAGE, GIN, GAT and GATv2 graph classifiers are intended to share

    - The functions remain in experiments/train.py for now because only the GCN currently uses them

        - Shared training and evaluation source files will be introduced when genuine reuse begins in Stage 3

        - This avoids creating abstractions before they are needed while also avoiding a GCN-specific training design that would later need to be rewritten

- Added one settings dictionary containing the choices that define this development fit

    - dataset="MUTAG" selects the dataset used by the run

    - baseline_width=64 gives the fixed GCN representation width

    - learning_rate=0.01 gives the Adam learning rate used for the GCN

        - The learning rate controls the scale of parameter updates made during optimisation

    - weight_decay=0.0005 gives the fixed Adam weight-decay setting

        - Weight decay penalises large parameter values during optimisation and acts as a form of regularisation

    - max_epochs=1000 gives the maximum number of complete training passes

    - batch_size=32 requests up to 32 graphs in each minibatch

    - seed=0 controls stochastic behaviour during fitting

    - split_seed=0 reconstructs the development partition established in Stage 1

    - Ordinary unchanged Adam options are left as PyTorch defaults rather than being presented as additional experimental settings

- Added main as the experiment-running part of the file

    - main performs the sequence required to conduct this particular development run

        - It loads the dataset

        - It obtains the development split

        - It sets the training seed

        - It creates the DataLoaders

        - It selects the device

        - It creates the GCN and Adam optimiser

        - It runs the training and validation epochs

    - The __name__ condition calls main when experiments.train is executed as a module

        - This means importing the module elsewhere does not automatically begin a 1,000-epoch experiment

- Reused load_dataset to load MUTAG using the same dataset representation established in Stage 1

    - The GCN therefore receives the same 7 categorical node-feature channels and graph connectivity already inspected earlier

    - Edge features remain excluded from the principal model input

- Reused stratified_split to reconstruct the development partitions

    - train_indices contains the graphs used for parameter updates

    - val_indices contains the graphs used for validation

    - The returned development test indices are assigned to _ because they are intentionally unused in 2.3

    - split_seed and seed perform different roles

        - split_seed determines which original graphs belong to the training, validation and test partitions

        - seed controls stochastic behaviour during fitting without changing those memberships

- Seeded the stochastic fitting process

    - set_seed(seed) seeds the Python, NumPy and PyTorch random-number generators as established in Stage 1

    - A separate torch.Generator is also created and seeded for the training DataLoader

    - The generator controls the random graph ordering produced by shuffle=True

        - This makes the shuffled training order reproducible

        - Keeping the loader's generator explicit also avoids making its ordering depend unnecessarily on random-number consumption during construction of a particular model

- Created separate training and validation DataLoaders

    - The training loader contains only dataset[train_indices]

    - batch_size=32 requests minibatches containing up to 32 graphs

    - shuffle=True changes the order in which training graphs are grouped and presented across epochs

    - The validation loader contains only dataset[val_indices]

    - shuffle=False keeps validation ordering fixed because validation does not optimise the model

- Confirmed the development partition and minibatch counts during execution

    - Training graphs: 150

    - Validation graphs: 18

    - Training batches: 5

        - Four training minibatches contain 32 graphs each

        - The remaining 22 graphs form the final smaller minibatch

        - 32 + 32 + 32 + 32 + 22 = 150

    - Validation batches: 1

        - All 18 validation graphs fit into one batch because the batch size is 32

    - These counts match the development split established in Stage 1

- Added automatic PyTorch device selection

    - torch.cuda.is_available() checks whether CUDA can currently be used

    - torch.device selects CUDA when available and otherwise falls back to CPU

    - model.to(device) moves the model parameters to the selected device

    - graph_batch.to(device) moves the tensors in each PyG minibatch to the same device before model operations are performed

    - Execution reported Device: cuda

        - The development fit therefore ran using the available CUDA GPU

- Created an Adam optimiser for the model parameters

    - model.parameters() supplies all registered trainable tensors in the two GCN layers and the classifier

    - The complete GCN contained 4,802 trainable scalar parameters in 2.2

    - Adam uses gradients of the loss to determine updates to those parameter values

    - The project-selected learning rate of 0.01 and weight decay of 0.0005 are supplied explicitly

    - Other unchanged Adam behaviour remains at the normal PyTorch defaults

- Added train_epoch to perform one complete optimisation pass through the training partition

    - An epoch is one complete pass through all 150 training graphs

    - model.train() places the model in training mode before minibatches are processed

        - The current GCN does not contain dropout or batch-normalisation layers whose behaviour changes between training and evaluation modes

        - Using the correct mode still gives the training function the standard PyTorch behaviour expected by models where such a distinction matters

    - Each graph_batch is moved to the selected device before the forward pass

- Implemented the ordinary minibatch training sequence explicitly

    - optimizer.zero_grad() clears parameter gradients remaining from the previous minibatch

        - PyTorch accumulates gradients by default

        - Without resetting them, gradients from previous minibatches would be unintentionally added to the current gradients

    - Calling model with graph_batch.x, graph_batch.edge_index and graph_batch.batch performs the forward pass

        - graph_batch.x contains the node features

        - graph_batch.edge_index supplies the graph connectivity used by the GCN layers

        - graph_batch.batch identifies which graph each node belongs to for global sum pooling

        - The returned logits contain one row for each graph and one raw score for each graph class

    - F.cross_entropy compares the logits with graph_batch.y

        - graph_batch.y contains the correct class index for every graph in the minibatch

        - Cross-entropy accepts raw logits directly, so the model does not apply softmax before the loss

        - The loss is lower when the model's class scores provide stronger relative support for the correct targets

        - F.cross_entropy returns the mean loss across the graphs in the current minibatch

    - loss.backward() performs backpropagation

        - PyTorch follows the operations that produced the loss and calculates gradients with respect to each trainable model parameter

        - A gradient describes how changing a parameter would locally affect the loss

        - backward calculates these gradients but does not itself change the parameter values

    - optimizer.step() performs the Adam parameter update

        - Adam uses the calculated gradients and its internal optimiser state to change the model parameters

    - This zero-gradient, forward, loss, backward and update sequence is repeated for each of the five training minibatches in an epoch

- Added class predictions and accuracy

    - logits.argmax(dim=1) selects the index of the largest class logit in each graph's output row

    - dim=1 is the class dimension of the graph-by-class logits tensor

    - The resulting predicted class indices are compared with graph_batch.y

    - Correct predictions are accumulated across all minibatches

    - Accuracy is calculated as the total number of correctly classified graphs divided by the total number of graphs

- Calculated loss as a mean over individual graphs rather than an unweighted mean over minibatches

    - The cross-entropy value returned for one minibatch is already a mean over the graphs in that minibatch

    - The final training minibatch contains 22 graphs while the other four contain 32

    - Simply averaging the five minibatch mean losses would therefore give each graph in the smaller final minibatch more influence than a graph in a full minibatch

    - graph_batch.num_graphs gives the number of graphs in the current minibatch

    - loss.item() extracts the scalar numerical minibatch loss for metric reporting

    - The minibatch mean is multiplied by its graph count before being added to total_loss

        - This recovers that minibatch's contribution to the total loss across individual graphs

    - The accumulated loss is divided by total_graphs at the end of the partition

        - This produces the mean cross-entropy per graph over the full training or validation partition

    - Accuracy follows the same graph-level principle by accumulating correct predictions and graph counts rather than averaging batch accuracies

- Added evaluate to measure the model on validation graphs without fitting to them

    - model.eval() places the model in evaluation mode

    - torch.no_grad() disables gradient tracking during validation

        - Validation does not require backpropagation because its purpose is measurement rather than parameter optimisation

        - Disabling gradient tracking avoids constructing unnecessary gradient information

    - The same forward pass, cross-entropy definition and prediction rule are used during validation

        - Training and validation loss therefore have the same mathematical meaning

        - Training and validation accuracy also use the same definition

    - Validation does not call optimizer.zero_grad(), loss.backward() or optimizer.step()

        - The validation graphs therefore do not directly update the model parameters

- Added the ordinary epoch loop

    - Epochs are numbered from 1 through the fixed maximum of 1,000

    - train_epoch first performs one optimisation pass through the complete training partition

    - evaluate then measures the resulting parameter state on the validation partition

    - Training loss, training accuracy, validation loss and validation accuracy are printed every ten epochs

    - No early stopping or best-model restoration occurs in 2.3

        - The purpose of this substage is to establish ordinary fitting and validation before introducing model selection

        - Validation-selected state preservation and patience are introduced in 2.4

- Ran python -m experiments.train successfully for all 1,000 epochs

    - The fit completed without a training, device, loss, gradient or validation error

    - Training loss at the printed checkpoints generally decreased over the run

        - It was 0.5014 at epoch 10

        - It was 0.3861 at epoch 100

        - It was 0.3174 at epoch 500

        - It was 0.2938 at epoch 1,000

        - Individual checkpoints still fluctuated because minibatch training with shuffled data is stochastic

    - Training accuracy generally increased

        - It was 0.6867 at epoch 10, corresponding to about 103 correct predictions out of 150

        - It reached 0.8533 at epoch 100

        - It reached 0.8867 at epoch 500

        - It was 0.8667 at epoch 1,000

        - The highest accuracy among the printed ten-epoch checkpoints was 0.9067 at epoch 740

        - Training accuracy is not required to improve monotonically because the parameters continue to change through stochastic minibatch updates

- Validation behaviour differed from the general training-loss trend

    - Validation loss initially improved substantially

        - It was 0.4496 at epoch 10

        - It was 0.3294 at epoch 100

        - It was 0.2772 at epoch 200

    - The lowest validation loss visible among the printed ten-epoch checkpoints was 0.2594 at epoch 250

        - This must not be called the actual best epoch because metrics were only printed every ten epochs

        - An unprinted epoch between checkpoints may have had a lower validation loss

        - Stage 2.4 will inspect validation loss after every epoch and preserve the actual selected state

    - After the earlier improvement, validation loss became more variable and often increased while training loss remained relatively low

        - Examples include validation loss 0.3407 at epoch 560, 0.3789 at epoch 680, 0.3730 at epoch 830 and 0.3867 at epoch 890

        - This demonstrates why the final state at epoch 1,000 should not automatically be assumed to be the preferred model state

        - It also motivates validation-based model-state selection and early stopping in 2.4

- Validation accuracy was much more discrete than validation loss

    - There are only 18 validation graphs

    - One changed graph prediction changes validation accuracy by 1 / 18, which is approximately 0.0556

    - The commonly observed validation accuracies therefore included 0.8333, 0.8889 and 0.9444

        - 0.8333 corresponds to 15 correct graphs out of 18

        - 0.8889 corresponds to 16 correct graphs out of 18

        - 0.9444 corresponds to 17 correct graphs out of 18

    - Validation accuracy sometimes changed sharply even when validation loss changed more gradually

        - This illustrates why loss provides more information about the model's class scores than the small validation set's discrete correct/incorrect count alone

    - The project will therefore use minimum validation cross-entropy rather than validation accuracy to select model states in 2.4

- The 2.3 run establishes that the complete ordinary supervised-learning path works end to end

    - MUTAG graphs were loaded and split correctly

    - Training graphs were shuffled reproducibly and processed in minibatches

    - The model ran on CUDA

    - Graph-level logits were compared with graph targets using cross-entropy

    - Backpropagation produced gradients and Adam updated the model over repeated epochs

    - Graph-weighted training loss and accuracy were calculated

    - Validation loss and accuracy were measured without validation parameter updates

    - The full fixed maximum of 1,000 epochs completed successfully

    - The run does not establish the final development-test performance or the correct model-selection epoch

        - Those questions depend on the validation-selected state implemented in 2.4

- Commit: 2.3 added ordinary fitting and validation





# 2.3 Ordinary Fitting and Validation

- Added experiments/train.py to introduce ordinary supervised fitting and validation

    - The complete GCN from 2.1 and 2.2 can now be fitted to labelled MUTAG graphs rather than only inspected with untrained parameters

    - Training uses the development training partition to update model parameters

    - Validation measures the current model after every training epoch without directly updating its parameters

    - The development test partition remains unused in the final 2.3 implementation

        - Test assessment will only be introduced after validation-based model-state selection and restoration are implemented in 2.4

- Kept the training implementation simple while making the natural shared parts suitable for the later GNN models

    - train_epoch receives the model as an argument rather than containing GCN-specific layer operations

    - evaluate also receives the model as an argument

    - Both functions only rely on the common graph-classification interface

        - The model receives x, edge_index and batch

        - The model returns one row of graph-level class logits for every graph in the minibatch

    - GraphSAGE, GIN, GAT and GATv2 are intended to expose the same basic interface

    - The functions remain in experiments/train.py for now because only one model currently uses them

        - Shared training and evaluation modules will be introduced when multiple models create a genuine reuse requirement

        - This avoids both GCN-specific duplication and unnecessary abstraction before it is needed

- Added one settings dictionary containing the choices that define this development fit

    - dataset=MUTAG selects the development dataset

    - baseline_width=64 gives the fixed representation width of the contextual GCN

    - learning_rate=0.01 gives the Adam learning rate

        - The learning rate influences the size of the parameter updates made during optimisation

    - weight_decay=0.0005 gives the fixed Adam weight-decay setting

        - Weight decay discourages excessively large parameter values during optimisation

    - epochs=1000 gives the fixed number of complete training epochs

        - The setting is called epochs rather than max_epochs because the current procedure deliberately completes all 1,000 epochs

        - No patience or early-stopping rule is used

    - batch_size=32 requests up to 32 graphs in each minibatch

    - seed=0 controls stochastic behaviour during model fitting

    - split_seed=0 reproduces the fixed development partition established in Stage 1

    - Ordinary unchanged Adam defaults are left as PyTorch defaults rather than being added to the settings as if they were additional experimental choices

- Added main as the experiment-running part of the file

    - main performs the operations required for this particular development fit

        - It loads MUTAG

        - It obtains the fixed training and validation indices

        - It sets the training seed

        - It creates the DataLoaders

        - It selects the compute device

        - It constructs the GCN

        - It constructs the Adam optimiser

        - It runs all training and validation epochs

    - The __name__ condition calls main when experiments.train is executed as a module

        - This prevents a full training run from beginning merely because the module is imported elsewhere

- Reused load_dataset to load MUTAG consistently with Stage 1

    - The model therefore uses the same 7 loaded categorical node-feature channels previously inspected

    - edge_index continues to provide graph connectivity

    - Edge features remain excluded from the principal model input

- Reused stratified_split to reconstruct the fixed development training and validation partitions

    - train_indices identifies the graphs used for parameter updates

    - val_indices identifies the graphs used for validation

    - The third returned partition is assigned to _ because development-test evaluation does not belong in ordinary fitting and validation

    - split_seed and seed continue to have different purposes

        - split_seed determines which original graphs belong to each development partition

        - seed controls stochastic behaviour during fitting without changing graph membership

- Seeded the stochastic fitting process

    - set_seed seeds the Python, NumPy and PyTorch random-number generators

    - A separate torch.Generator is created for the shuffled training DataLoader

    - manual_seed gives this generator the same training seed

    - Supplying the generator to the DataLoader makes the shuffled training order reproducible

    - Keeping loader shuffling explicitly seeded also avoids making its random sequence depend unnecessarily on how much randomness a particular model consumes during construction

        - This is useful when several GNN models later share the same fitting machinery

- Created separate training and validation DataLoaders

    - The training loader receives dataset[train_indices]

    - batch_size=32 requests up to 32 graphs per training minibatch

    - shuffle=True changes the graph ordering used to form training minibatches across epochs

    - The validation loader receives dataset[val_indices]

    - shuffle=False is used because validation measures the model rather than optimising it

- Confirmed the expected development partition sizes during execution

    - Training graphs: 150

    - Validation graphs: 18

    - These match the Stage 1 development split

- Confirmed the expected minibatch counts

    - Training batches: 5

        - Four full training minibatches contain 32 graphs each

        - The remaining 22 graphs form the fifth minibatch

        - 32 + 32 + 32 + 32 + 22 = 150

    - Validation batches: 1

        - All 18 validation graphs fit into one minibatch because the requested batch size is 32

- Added automatic device selection

    - torch.cuda.is_available checks whether CUDA is available to PyTorch

    - torch.device selects CUDA when it is available and otherwise falls back to CPU

    - model.to(device) moves the model parameters to the selected device

    - graph_batch.to(device) moves the tensors in each PyG minibatch to the same device before model computation

    - Execution reported Device: cuda

        - The 1,000-epoch development fit therefore ran using the available CUDA device

- Created the Adam optimiser

    - model.parameters supplies the trainable tensors registered by the two GCN layers and graph classifier

    - The complete GCN contained 4,802 trainable scalar parameters in 2.2

    - Adam uses gradients of the loss to update these parameter values

    - learning_rate=0.01 and weight_decay=0.0005 are supplied because they are actual project choices

    - The remaining unchanged Adam behaviour uses the ordinary PyTorch defaults

- Added train_epoch for one complete optimisation pass through the training partition

    - One epoch means processing every one of the 150 training graphs once

    - model.train places the model into training mode before processing the minibatches

        - The current GCN does not contain dropout or batch-normalisation layers whose numerical behaviour changes between training and evaluation modes

        - Using the appropriate mode still gives the fitting function the standard PyTorch behaviour required by models where the distinction matters

    - Every graph_batch is moved to the selected device before model operations are performed

- Implemented the ordinary minibatch optimisation sequence explicitly

    - optimizer.zero_grad clears parameter gradients remaining from the previous minibatch

        - PyTorch accumulates gradients by default

        - Clearing them ensures that the following update uses the gradients calculated from the current minibatch rather than unintentionally combining them with previous gradients

    - Calling the model performs the forward pass

        - graph_batch.x supplies node features

        - graph_batch.edge_index supplies graph connectivity

        - graph_batch.batch identifies which graph each node belongs to for sum pooling

        - The complete GCN produces one row of raw class logits for every graph in the minibatch

    - F.cross_entropy compares these logits with graph_batch.y

        - graph_batch.y contains the correct graph-class index for each graph

        - Cross-entropy accepts the raw logits directly, so the model does not apply softmax before calculating the loss

        - The returned value is the mean classification loss across the graphs in the current minibatch

    - loss.backward performs backpropagation

        - PyTorch traces backwards through the operations that produced the loss

        - It calculates a gradient for each trainable parameter describing how the loss locally changes with that parameter

        - backward calculates gradients but does not itself change the model parameters

    - optimizer.step performs the Adam parameter update

        - Adam uses the calculated gradients and its optimiser state to change the trainable parameter values

    - This zero-gradient, forward, loss, backward and update sequence occurs for each of the five training minibatches in every epoch

- Added graph-level predictions and accuracy

    - logits.argmax(dim=1) selects the index of the largest class logit for each graph

    - dim=1 is the class dimension of the graph-by-class output tensor

    - These predicted class indices are compared with graph_batch.y

    - The number of correct predictions is accumulated across all minibatches

    - Accuracy is calculated as the total number of correctly classified graphs divided by the total number of graphs

- Calculated partition loss as a mean over graphs rather than an unweighted mean of minibatch means

    - F.cross_entropy returns the mean loss within the current minibatch

    - The final training minibatch contains only 22 graphs while the preceding minibatches contain 32

    - Giving all five minibatch means equal weight would therefore give each graph in the smaller final minibatch more influence

    - graph_batch.num_graphs gives the number of graphs in the current minibatch

    - loss.item extracts the scalar numerical minibatch loss used for metric reporting

    - Multiplying the minibatch mean by num_graphs gives that minibatch's total contribution to the graph losses

    - These contributions are accumulated across the partition

    - Dividing by total_graphs produces the mean cross-entropy per graph over the complete partition

    - Accuracy follows the same graph-level principle by accumulating correct predictions and total graph counts instead of averaging minibatch accuracies

- Added evaluate for validation

    - evaluate measures loss and accuracy without modifying the model parameters

    - model.eval places the model into evaluation mode

    - torch.no_grad disables gradient tracking while validation calculations are performed

        - Validation does not require backpropagation because it performs no parameter update

        - Avoiding gradient tracking also avoids unnecessary gradient-related computation and storage

    - Validation uses the same model forward pass as training

    - The same cross-entropy definition and graph-level prediction rule are used

        - Training loss and validation loss therefore have the same mathematical meaning

        - Training accuracy and validation accuracy also use the same definition

    - Validation does not call optimizer.zero_grad, loss.backward or optimizer.step

        - The validation graphs therefore measure the current fitted model but do not directly train it

- Added the fixed 1,000-epoch training loop

    - Epoch numbers run from 1 through 1,000

    - train_epoch first performs a complete optimisation pass through the 150 training graphs

    - evaluate then measures the resulting model on the 18 validation graphs

    - Training loss, training accuracy, validation loss and validation accuracy are printed every ten epochs

    - Validation is still calculated after every epoch even though only every tenth set of metrics is printed

        - This distinction will matter in 2.4 when validation loss is used to select a model state

    - No patience or early stopping is used

        - Every fit deliberately completes the same fixed number of epochs

        - 2.4 will select the lowest-validation-loss state from within those epochs rather than using validation to decide when fitting stops

- Ran python -m experiments.train successfully for all 1,000 epochs

    - The run completed without a loading, device, forward-pass, loss, backpropagation, optimiser or validation error

    - The settings printed by the final run were:

        - dataset: MUTAG

        - baseline_width: 64

        - learning_rate: 0.01

        - weight_decay: 0.0005

        - epochs: 1000

        - batch_size: 32

        - seed: 0

        - split_seed: 0

- Training loss generally decreased over the course of fitting

    - At epoch 10, training loss was 0.5014

    - At epoch 100, training loss was 0.3849

    - At epoch 500, training loss was 0.3200

    - At epoch 1,000, training loss was 0.2928

    - The decrease was not monotonic

        - For example, training loss rose to 0.3969 at epoch 210 and to 0.3566 at epoch 480 before falling again

        - Such fluctuations are compatible with minibatch optimisation on shuffled data because each update is based on only part of the training partition

- Training accuracy generally increased compared with the beginning of the run

    - At epoch 10, training accuracy was 0.6867

        - With 150 training graphs, this corresponds to approximately 103 correctly classified training graphs

    - At epoch 100, training accuracy was 0.8400

    - At epoch 500, training accuracy was 0.8733

    - At epoch 1,000, training accuracy was 0.8667

    - The highest training accuracy among the printed ten-epoch checkpoints was 0.9067 at epoch 740

    - Training accuracy also fluctuated rather than increasing at every checkpoint because parameter updates continued throughout the fit

- Validation loss initially improved substantially but later became more variable

    - Validation loss was 0.4496 at epoch 10

    - It had fallen to 0.3296 by epoch 100

    - It was 0.2772 at epoch 200

    - Among the printed ten-epoch checkpoints, the lowest visible validation loss was 0.2633 at epoch 250

    - The printed checkpoints do not establish that epoch 250 was the actual minimum-validation-loss epoch

        - Validation was calculated after every epoch but only every tenth result was printed

        - One of the unprinted epochs may have produced a lower validation loss

        - 2.4 will explicitly track validation loss after every epoch and record the actual selected epoch

- Validation loss later increased while training loss generally remained lower

    - Validation loss was 0.3226 at epoch 540

    - It was 0.3562 at epoch 680

    - It reached 0.3765 at epoch 830

    - It reached 0.3949 at epoch 960

    - It was 0.3354 at epoch 1,000

    - The epoch-1,000 validation loss was therefore substantially higher than the best values visible earlier in training

    - This establishes why simply using the parameter state from the final epoch would not be an appropriate model-selection rule

    - It motivates 2.4, where the lowest-validation-loss state across the fixed 1,000 epochs will be retained and restored

- Validation accuracy was highly discrete because the validation partition contains only 18 graphs

    - Changing the prediction for one validation graph changes accuracy by 1 / 18, approximately 0.0556

    - The frequently observed validation accuracies therefore included:

        - 0.8333, corresponding to 15 correct predictions out of 18

        - 0.8889, corresponding to 16 correct predictions out of 18

        - 0.9444, corresponding to 17 correct predictions out of 18

    - Validation accuracy sometimes changed between these values while validation loss changed in a different direction

        - Accuracy records only whether each final class prediction is correct

        - Cross-entropy also reflects the relative class scores assigned by the model

    - Validation loss will therefore be used for model-state selection rather than choosing whichever epoch happened to have the highest validation accuracy

- The final 2.3 implementation deliberately does not return or report a selected epoch

    - Its purpose is to establish ordinary fitting and validation first

    - The final in-memory model after this substage is simply the state produced after epoch 1,000

    - No claim is made that this final state is the preferred model state

    - 2.4 will add explicit tracking of:

        - the epoch with the lowest validation loss

        - that epoch's validation loss

        - that epoch's validation accuracy

        - a preserved copy of that epoch's model state

    - After all 1,000 epochs have completed, 2.4 will restore that validation-selected state before any development-test assessment

- The 2.3 run establishes that the ordinary supervised GCN fitting path works end to end

    - The fixed development split was reconstructed correctly

    - Training minibatches were shuffled reproducibly

    - The model and graph tensors ran on CUDA

    - Raw graph logits were compared with graph labels using cross-entropy

    - Backpropagation calculated parameter gradients

    - Adam updated the GCN and classifier parameters

    - Training loss and accuracy were calculated over individual graphs

    - Validation loss and accuracy were calculated without parameter updates

    - All 1,000 fixed epochs completed successfully

    - Model-state selection and restoration remain deliberately unimplemented until 2.4

- Commit: 2.3 added ordinary fitting and validation





# 2.4 Validation-Selected State and Restoration

- Extended the fixed 1,000-epoch fitting procedure with validation-based model-state selection

    - Ordinary fitting in 2.3 evaluated the validation partition after every epoch but did not preserve a particular model state

    - The model remaining after 2.3 was simply the state produced by epoch 1,000

    - The 2.3 validation trajectory showed that earlier epochs could have substantially lower validation loss than the final epoch

    - 2.4 therefore selects the state with the lowest validation loss across the complete fit rather than automatically assessing the final training state

- Kept the fixed training length of 1,000 epochs

    - No patience or early stopping is used

    - Every fit completes exactly the number of epochs specified by settings["epochs"]

    - Validation determines which fitted state is retained but does not determine when fitting stops

    - This keeps the optimisation budget fixed while allowing model selection to choose an earlier state

- Added train_model to contain fixed-epoch fitting and validation-state selection

    - train_model receives the model, training loader, validation loader, optimiser, device and number of epochs

    - It reuses train_epoch for parameter optimisation

    - It reuses evaluate for validation measurement

    - The function does not depend on GCN-specific layers

        - It only requires a graph classifier that accepts x, edge_index and batch and returns graph-level logits

        - The same fitting procedure can therefore later be reused by GraphSAGE, GIN, GAT and GATv2

    - The function remains in experiments/train.py while only one model currently uses it

- Added best_epoch to record the selected epoch

    - best_epoch starts at 0 before any model state has been evaluated

    - Validation loss is inspected after every completed training epoch

    - Whenever validation loss becomes strictly lower than every value previously observed, best_epoch is replaced with the current epoch number

    - Selection is performed every epoch even though progress is only printed every ten epochs

        - The selected epoch therefore does not need to appear in the printed progress output

- Added best_val_loss as the model-selection criterion

    - best_val_loss starts at positive infinity using float("inf")

    - The first finite validation loss is therefore guaranteed to become the initial selected value

    - A later epoch replaces it only when val_loss < best_val_loss

    - The strict less-than comparison means an exactly equal validation loss does not replace the existing selected state

        - The earlier epoch is therefore retained in an exact tie

    - Validation cross-entropy rather than validation accuracy determines which model state is selected

- Added best_val_accuracy to record the accuracy belonging to the selected state

    - best_val_accuracy is only updated when validation loss strictly improves

    - It therefore records the validation accuracy produced by the same epoch as best_val_loss

    - Validation accuracy does not independently choose another epoch

    - best_epoch, best_val_loss and best_val_accuracy consequently describe one consistent model state

- Added preservation of the selected model state using PyTorch tensor cloning

    - model.state_dict() provides the current model state as named parameter and buffer tensors

    - These tensors contain the numerical state reached at the current training epoch

    - Training continues after an epoch becomes selected, so the current model parameters continue to change

    - Recording only the selected epoch number would therefore not preserve the model produced by that epoch

    - When validation loss strictly improves, every tensor in model.state_dict() is copied using value.clone()

    - clone creates a new PyTorch tensor containing the same numerical values with independent tensor storage

        - Later optimiser updates to the current model therefore do not alter the cloned values stored in best_state

    - A dictionary comprehension retains the original state-dictionary names while cloning every tensor

    - Only one selected state is retained at a time

        - If a later epoch improves validation loss, its cloned state replaces the previous best_state

        - A separate model is therefore not stored for every one of the 1,000 epochs

    - Using clone keeps the state preservation entirely within PyTorch and avoids requiring copy or deepcopy

- Restored the validation-selected state after all 1,000 epochs

    - The training loop always continues through epoch 1,000

    - At the end of the loop, the model initially contains the parameters produced by the final training epoch

    - model.load_state_dict(best_state) replaces those values with the cloned parameters from the minimum-validation-loss epoch

    - The model used after train_model therefore corresponds to the selected validation state rather than automatically to epoch 1,000

- Added selected-state statistics as return values from train_model

    - best_epoch gives the epoch at which the minimum validation loss occurred

    - best_val_loss gives that epoch's graph-mean validation cross-entropy

    - best_val_accuracy gives the validation accuracy produced by that same state

    - train_model returns these three values after restoring best_state

    - The model itself does not need to be separately returned because load_state_dict changes the existing model object in place

- Kept restoration handling minimal

    - No duplicate validation evaluation is performed immediately after restoring best_state

    - The selected validation loss and accuracy were already measured and stored when the selected state was originally encountered

    - The permanent implementation therefore focuses only on selecting, preserving, restoring and subsequently assessing the selected model

- Added the development test DataLoader

    - stratified_split now retains test_indices rather than discarding the third development partition

    - dataset[test_indices] selects the 20 development-test graphs established in Stage 1

    - batch_size remains 32

    - shuffle=False is used because the development test is evaluated rather than trained

    - All 20 development-test graphs therefore fit into one minibatch

- Added one preliminary development-test assessment after model selection

    - The test loader is not supplied to train_model

        - Development-test graphs therefore cannot influence gradients, parameter updates, validation loss or selection of the model epoch

    - Test evaluation occurs only after the complete 1,000-epoch fit has finished and the validation-selected state has been restored

    - The existing evaluate function is reused

        - model.eval() places the model in evaluation mode

        - torch.no_grad() disables gradient tracking

        - No parameter updates occur

    - The output is labelled Preliminary development test

        - This holdout provides an initial assessment of the working development procedure

        - It is not treated as the final project performance estimate

- Ran python -m experiments.train successfully for the complete 1,000-epoch fit

    - The run used the same fixed settings as 2.3:

        - dataset: MUTAG

        - baseline_width: 64

        - learning_rate: 0.01

        - weight_decay: 0.0005

        - epochs: 1000

        - batch_size: 32

        - seed: 0

        - split_seed: 0

    - Execution used CUDA

    - The development partition contained:

        - 150 training graphs

        - 18 validation graphs

        - 20 development-test graphs

- Training continued through epoch 1,000 as intended

    - There was no patience counter or early-stopping condition

    - The final printed epoch had:

        - Training loss: 0.2937

        - Training accuracy: 0.8800

        - Validation loss: 0.3095

        - Validation accuracy: 0.9444

    - The epoch-1,000 validation loss was not the lowest validation loss encountered during training

        - This confirms that the final model state and the validation-selected model state were different

- The selected model occurred at epoch 337

    - Selected epoch: 337

    - Selected validation loss: 0.2589

    - Selected validation accuracy: 0.9444

    - A validation accuracy of 0.9444 on 18 graphs corresponds to 17 correct graph predictions out of 18

    - Epoch 337 did not appear in the terminal's ten-epoch progress output

        - This directly confirms that validation-state selection was being performed after every epoch rather than only at printed checkpoints

        - Restricting selection to multiples of ten would therefore have missed the actual selected state

- The selected validation loss was substantially lower than the loss at the final epoch

    - Selected validation loss at epoch 337: 0.2589

    - Validation loss at epoch 1,000: 0.3095

    - The difference demonstrates why the epoch-1,000 parameter state should not automatically be used for assessment

    - Restoring the cloned epoch-337 state ensures that development-test evaluation uses the model chosen by the validation criterion

- The selected epoch was not simply the epoch with an unusually high validation accuracy

    - Validation accuracy of 0.9444 appeared at many other epochs during training

    - Only epoch 337 achieved the minimum validation loss of 0.2589 across the complete 1,000-epoch fit

    - This illustrates the distinction between the selection criterion and the additional statistic recorded for the selected state

        - Validation loss determines which state is selected

        - Validation accuracy describes the selected state once it has been chosen

- The restored selected model was evaluated once on the development test partition

    - Preliminary development-test loss: 0.5301

    - Preliminary development-test accuracy: 0.7500

    - With 20 development-test graphs, an accuracy of 0.7500 corresponds to 15 correctly classified graphs and 5 incorrectly classified graphs

    - This test result is lower than the selected validation accuracy of 0.9444

        - The validation partition contains only 18 graphs and the development-test partition contains only 20, so both are small samples

        - The difference is an observed development result rather than a reason to change the fixed training or model-selection procedure

    - The test result was obtained only after validation-based selection had already been completed

        - It therefore did not influence which epoch or parameter state was retained

- The completed 2.4 procedure now distinguishes fitting, selection and assessment

    - Training data determines parameter updates

    - Validation loss selects one state from the 1,000 states produced during fitting

    - PyTorch clone preserves that selected state independently of later optimisation

    - load_state_dict restores the selected state after the fixed training budget has finished

    - The selected epoch and its validation loss and accuracy are returned explicitly

    - The development test then provides one preliminary assessment of that restored state

- Stage 2.4 therefore establishes the complete validation-selected development fitting procedure required before result recording

    - The model is no longer assessed simply because it is the final epoch

    - Selection uses every epoch rather than only printed checkpoints

    - Selected-state preservation is independent of subsequent optimiser updates

    - No patience or early stopping is used

    - Every fit receives the same fixed 1,000-epoch training budget

    - The selected epoch and its statistics are available for later result recording

- Commit: 2.4 added validation state selection and restoration





# 2.5 Result Recording and Source Provenance

- Added src/recording.py to preserve development experiment evidence as JSON

    - The recorder remains a small collection of ordinary functions rather than introducing an experiment-management framework

    - get_source_commit runs git rev-parse HEAD through subprocess to obtain the exact Git commit containing the source used for the fit

        - subprocess is needed because the commit identity comes from Git rather than Python itself

        - The entire repository is not required to have a clean working tree

        - Instead, the experiment source and effective settings used by the fit are committed before the recorded run

    - get_result_path constructs a descriptive filename from the result purpose, model, dataset, optional variant and training seed

        - The current result is therefore stored as results/development_gcn_mutag_seed0.json

    - prepare_result_path creates the results directory when it does not already exist

        - os.makedirs with exist_ok=True creates the directory when needed and leaves an existing directory unchanged

        - The same function checks whether the intended result file already exists before training starts

        - This prevents an expensive fit from being run when its intended result filename is already occupied

    - save_result writes the JSON using file mode x

        - Mode x refuses to replace an existing file

        - The pre-fit path check and the write-time protection together prevent silent loss of earlier experiment evidence

    - save_result also records the PyTorch, PyG and CUDA versions automatically

- Added model and variant identity to the effective settings

    - model=GCN is now used both as recorded experiment information and when constructing the result filename

    - variant=None records that this ordinary GCN has no special experimental variant

        - Python None becomes null when written as JSON

    - The remaining effective settings are:

        - dataset=MUTAG

        - baseline_width=64

        - learning_rate=0.01

        - weight_decay=0.0005

        - epochs=1000

        - batch_size=32

        - seed=0

        - split_seed=0

    - Ordinary unchanged Adam defaults remain library defaults rather than being added as experimental factors

- Added source provenance before the recorded fit

    - The successful fit records source commit 9898a4a8d4f8da9baf5460dcc28ee95d7422eaba

    - This is the commit containing the experiment source and settings that produced the model

    - The result JSON and completed log are committed afterwards in a separate commit

    - The later result commit is therefore not confused with the source commit that produced the fit

- Added training-pass runtime measurement

    - time.perf_counter is used as the wall-clock timer

    - Only train_epoch is inside the timed region

    - The timed training pass therefore includes:

        - Iterating through the training DataLoader

        - Moving graph minibatches to the selected device

        - Forward computation

        - Cross-entropy calculation

        - Backpropagation

        - Adam parameter updates

    - Validation, development-test evaluation and result writing remain outside the timed region

    - CUDA work can execute asynchronously relative to Python, so torch.cuda.synchronize is called immediately before and after the timed region when CUDA is used

        - This makes the wall-clock boundaries wait for the measured GPU work to complete

- Kept the validation-selected state procedure from 2.4

    - Every fit completes the fixed 1,000-epoch training budget

    - No patience or early stopping is used

    - Validation is evaluated after every epoch

    - The state with the strictly smallest validation cross-entropy is selected

    - An exact validation-loss tie retains the earlier epoch

    - The selected model tensors are preserved with PyTorch clone and restored after epoch 1,000

- Added direct parameter counting to the experiment evidence

    - parameter.numel() gives the number of scalar values stored in one parameter tensor

    - Summing numel across all model parameters gives the total parameter count

    - Summing only parameters whose requires_grad value is True gives the active trainable parameter count

- Added the dataset loader and feature policy to the recorded result

    - cleaned=False records use of the ordinary rather than cleaned TU dataset

    - use_node_attr=False records that additional node attributes are not requested

    - use_edge_attr=False records that additional continuous edge attributes are not requested

    - MUTAG categorical node labels are still used as the node features supplied to the GCN

    - Graph connectivity is used through edge_index

    - Edge features are not supplied to the GCN

- Added the original development partition indices

    - The JSON contains the original graph indices for all three development partitions

    - 150 graphs are assigned to training

    - 18 graphs are assigned to validation

    - 20 graphs are assigned to the development test

    - These counts account for all 188 MUTAG graphs

    - Recording original indices preserves the identities of the graphs used in each role rather than only preserving partition sizes

- Added validation-selection evidence to the JSON

    - The criterion is recorded as minimum validation cross-entropy

    - The tie rule is recorded as earliest exact tie

    - early_stopping=False explicitly records the fixed-epoch procedure

    - All 1,000 epochs completed

    - Epoch 384 was selected

        - Its validation loss was 0.256849080324173

        - Its validation accuracy was 0.8333333333333334, corresponding to 15 of the 18 validation graphs

    - Some printed epochs reached higher validation accuracy, but they were not selected because validation loss rather than validation accuracy is the selection criterion

    - Epoch 384 is not one of the ten-epoch reporting points

        - This confirms that model-state selection is performed after every epoch while terminal progress is only printed every ten epochs

- Added the development-test result

    - The development test is evaluated only after the validation-selected state has been restored

    - Development-test loss was 0.49510836601257324

    - Development-test accuracy was 0.8, corresponding to 16 of the 20 development-test graphs

    - The result remains explicitly identified as development evidence rather than final cross-validation evidence

- Recorded model size and runtime

    - The GCN has 4,802 total parameters

    - All 4,802 parameters are trainable

        - This agrees with the parameter count established when the complete GCN was built in 2.2

    - The 1,000 timed training passes took 37.63074249937199 seconds in total

    - Mean timed training-pass duration was 0.03763074249937199 seconds per epoch, approximately 37.6 milliseconds

- Recorded execution environment

    - The run used CUDA on an NVIDIA GeForce RTX 4070 Laptop GPU

    - PyTorch version was 2.13.0+cu130

    - PyG version was 2.8.0.post1

    - CUDA version was 13.0

    - Recording the device and relevant software versions provides the execution context needed for reproducibility and later runtime comparisons

- Inspected the actual development JSON

    - results/development_gcn_mutag_seed0.json contains the expected purpose, settings, loader policy, feature policy, partitions, validation-selection evidence, development-test metrics, parameter counts, runtime information, device information, software versions and source commit

    - The values in the JSON agree with the successful terminal execution

    - The recorded source commit exactly matches the commit printed before training began

- A recording failure exposed one missing filesystem requirement during development

    - An earlier 1,000-epoch attempt completed training but failed when saving because the results directory did not yet exist

    - prepare_result_path was therefore changed to create the results directory before training begins

    - The successful recorded fit then completed and saved the JSON normally

- Commit: 2.5 recorded the GCN development fit