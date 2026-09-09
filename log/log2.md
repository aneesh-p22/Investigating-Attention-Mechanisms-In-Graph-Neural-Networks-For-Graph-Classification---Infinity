# 2.1 GCN Layers

- Added src/models/gcn.py containing the first GNN model in the project

    - GCN inherits from torch.nn.Module

        - nn.Module is PyTorch's base class for neural-network models

        - Inheriting from it allows PyTorch to keep track of the model's registered layers and parameters

    - super().__init__() initialises the nn.Module part of the class so that layers assigned to the model are registered correctly

- Added two GCNConv message-passing layers

    - GCNConv is PyTorch Geometric's implementation of the Graph Convolutional Network operation

    - A GCN layer updates each node using its own representation and the representations of neighbouring nodes

        - Neighbour contributions are degree-normalised so that aggregation is scaled according to the connectivity of the nodes

        - A shared learned linear transformation maps node features into the requested output feature width

        - The layer combines this feature transformation with degree-normalised neighbourhood aggregation

    - The first layer maps the input feature width to 64

        - The original constructor called the input feature argument in_channels

        - MUTAG has seven loaded node features, so this layer receives a matrix with seven values per node

        - The layer produces 64 learned features for every node

    - The second layer maps 64 features to 64 features

        - It receives the 64-dimensional node representations produced by the first layer and ReLU

        - It produces another 64-dimensional representation for every node

    - Two message-passing layers mean that a final node representation can incorporate information propagated across up to two graph hops

- Used the fixed GCNConv settings required by the project

    - improved=False uses the standard GCN self-loop weighting rather than PyG's alternative improved formulation

    - cached=False means the normalised graph connectivity is recomputed for each input instead of being stored and reused

        - This is appropriate because training uses minibatches containing different graphs

    - add_self_loops=True includes a self-connection for each node inside GCNConv

        - This allows a node's own current representation to contribute to its next representation as well as its neighbours' representations

    - normalize=True applies the GCN symmetric degree normalisation to the connectivity

        - This corresponds to scaling the adjacency matrix using node degrees rather than simply summing all neighbouring features without adjustment

    - bias=True gives each output feature one additional learned bias value

- Added ReLU after both GCN layers

    - ReLU acts element by element, replacing negative values with zero while leaving positive values unchanged

    - ReLU introduces a nonlinearity between graph convolution operations

    - It changes feature values but does not change the tensor shape

    - ReLU has no trainable parameters of its own

    - The current model uses F.relu, while the inspector uses torch.relu; both apply the same activation

- Defined the model's forward method

    - forward describes the sequence of operations performed when the model is called

    - At this substage, it received x and edge_index

        - x contains one feature vector for every node in the minibatch

        - edge_index describes which nodes are connected and therefore which node representations can be exchanged during message passing

    - The sequence was first GCNConv, ReLU, second GCNConv, then ReLU

    - edge_attr was not passed to the model

        - MUTAG contains edge labels, but edge features are deliberately excluded from the principal models in this investigation

    - The forward method at this stage returned node embeddings

        - It did not yet combine nodes into graph representations or make class predictions

- Added experiments/models/inspect_gcn.py to inspect the new model on a real MUTAG minibatch

    - load_dataset("MUTAG") loads the same MUTAG representation established in Stage 1

    - DataLoader groups graphs into minibatches

        - batch_size=32 requests up to 32 graphs in one minibatch

        - shuffle=False keeps the dataset order fixed for this inspection

    - next(iter(loader)) retrieves the first minibatch produced by the DataLoader

    - The GCN is created with dataset.num_node_features as its input width

        - dataset.num_node_features is seven for the loaded MUTAG representation

        - The fixed contextual-baseline width is 64

- Inspected the model structure

    - Printing the model showed conv1 as GCNConv(7, 64)

    - Printing the model showed conv2 as GCNConv(64, 64)

    - These dimensions matched the intended two-layer architecture

- Inspected the registered parameters with model.named_parameters()

    - named_parameters() provides each registered parameter tensor together with its name

        - It does not exclude a parameter merely because requires_grad is False

        - All registered parameters in this GCN were trainable

    - parameter.shape shows the dimensions of a tensor

    - parameter.numel() gives the total number of scalar values contained in it

    - conv1.lin.weight had shape [64, 7]

        - PyTorch stores this linear weight tensor as output features by input features

        - It therefore contains 64 × 7 = 448 learned weights

    - conv1.bias had shape [64]

        - There is one learned bias for each of the 64 output features

        - The first GCN layer therefore contains 448 + 64 = 512 parameters

    - conv2.lin.weight had shape [64, 64]

        - It contains 64 × 64 = 4,096 learned weights

    - conv2.bias had shape [64]

        - The second GCN layer therefore contains 4,096 + 64 = 4,160 parameters

    - The two GCN layers contain 512 + 4,160 = 4,672 parameters

        - This matched the reported total of 4,672

        - All 4,672 parameters were trainable

- Inspected the tensor shapes through each model operation

    - The input x had shape [585, 7]

        - The first minibatch contained 32 graphs with 585 nodes in total

        - Each node had the seven loaded MUTAG node features

    - After conv1, the shape was [585, 64]

        - The number of nodes remained 585 because graph convolution updates node representations rather than combining or removing nodes

        - The feature width changed from seven to 64 because the first GCN layer produces 64 output features per node

    - After the first ReLU, the shape remained [585, 64]

        - ReLU changes values without changing dimensions

    - After conv2, the shape remained [585, 64]

        - There was still one representation for every node

        - The second layer retained the 64-feature width

    - After the second ReLU, the shape again remained [585, 64]

- Manually stepped through the layers before calling the complete model

    - Accessing model.conv1 and model.conv2 directly made the intermediate shape transitions visible

    - Calling model(graph_batch.x, graph_batch.edge_index) then executed the forward method normally

    - The complete model output had shape [585, 64]

        - This established successful execution with the expected output dimensions

        - Agreement between shapes did not independently establish numerical equality between the manual sequence and the complete model call

- The output at this stage was still node-level rather than graph-level

    - The minibatch contained 32 graphs, but the model returned 585 node vectors because no readout operation had yet been added

    - Stage 2.2 would use graph membership information to sum node vectors separately for each graph, producing one 64-dimensional representation per graph

- Execution

    - Ran python -m experiments.models.inspect_gcn successfully

    - The model structure, parameter counts and tensor dimensions matched the intended Stage 2.1 design

- 2.1 added and inspected GCN layers





# 2.2 Sum Pooling and Classifier

- Extended the GCN from a node-embedding model into a complete graph classifier

    - Stage 2.1 ended with one 64-dimensional representation for every node in the minibatch

    - Graph classification requires one representation for each whole graph

    - The completed forward path contains two GCNConv and ReLU blocks, global sum pooling and a linear classifier

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

    - edge_index and batch have different responsibilities

        - edge_index describes which nodes are connected and is used during GCN message passing

        - batch describes which whole graph each node belongs to and is used during graph-level pooling

- Sum pooling is permutation invariant

    - The numerical order assigned to the nodes of a graph is arbitrary and should not determine the graph representation

    - Mathematically, addition gives the same result when its terms are reordered

    - Reordering node representations within a graph therefore leaves their mathematical sum unchanged

    - This establishes invariance to node ordering

        - It does not imply that every pair of different graphs must produce different summed representations

- Added nn.Linear as the graph classifier

    - nn.Linear applies a learned affine transformation to each pooled graph representation

    - The classifier receives a graph representation with width 64

        - The original GCN constructor called this width argument baseline_width

        - The current constructor uses hidden_dim for the same role

    - The classifier produces num_classes output values for each graph

        - MUTAG has two graph classes, so the classifier maps each 64-dimensional graph representation to two output values

    - These two output values are logits

        - A logit is a raw score produced for one class

        - For MUTAG, each output row contains one score for class 0 and one score for class 1

        - The logits are not probabilities and do not need to sum to one

    - No softmax is applied inside the model

        - The cross-entropy loss introduced during training operates directly on these logits

- Updated the GCN constructor to receive num_classes

    - The input feature argument specifies the number of loaded features for each node

    - The hidden-width argument controls the 64-dimensional node and graph representations

    - num_classes determines the number of outputs produced by the classifier

    - The classifier is stored as self.classifier so PyTorch registers its weights and bias as model parameters

- Updated the GCN forward method to receive batch

    - x contains the feature representation for every node in the minibatch

    - edge_index contains the connectivity used by the two GCNConv layers

    - batch contains the graph membership of every node and is required by global_add_pool

    - The first GCNConv and ReLU produce the first set of learned node representations

    - The second GCNConv and ReLU produce the final node representations

    - global_add_pool sums those node representations separately for each graph

    - self.classifier converts every pooled graph representation into its class logits

    - The model now returns graph-level logits rather than node embeddings

- Updated experiments/models/inspect_gcn.py to inspect the complete graph classifier

    - The same first MUTAG minibatch of 32 graphs was used so that the new operations could be compared directly with the Stage 2.1 inspection

    - The model was constructed using dataset.num_classes in addition to the node-feature width and hidden width

        - dataset.num_classes is two for MUTAG

        - This makes the classifier output width match the two MUTAG target classes

    - The script manually applied the two GCN layers and ReLU operations before applying pooling and classification

        - This exposed the representation shape at each boundary of the complete model

    - The complete model was then called using graph_batch.x, graph_batch.edge_index and graph_batch.batch

        - This exercised the actual forward method after the individual operations had been inspected

- Inspected the updated model structure

    - The model contained GCNConv(7, 64) as conv1

        - This was unchanged from Stage 2.1

    - The model contained GCNConv(64, 64) as conv2

        - This was also unchanged from Stage 2.1

    - The model additionally contained Linear(in_features=64, out_features=2, bias=True) as classifier

        - in_features=64 matched the width of the pooled graph representation

        - out_features=2 matched the two MUTAG graph classes

        - bias=True meant the classifier learned one additional bias value for each output

- Inspected the classifier parameters

    - classifier.weight had shape [2, 64]

        - PyTorch stores the linear weight tensor as output features by input features

        - The classifier therefore contains 2 × 64 = 128 learned weights

    - classifier.bias had shape [2]

        - There is one learned bias value for each class logit

    - The classifier contains 128 + 2 = 130 parameters

    - The two GCN layers contain 4,672 parameters

    - Adding the classifier gives 4,672 + 130 = 4,802 parameters for the complete graph classifier

    - The reported values matched these calculations

        - Classifier parameters: 130

        - Total parameters: 4,802

    - All model parameters were trainable

    - global_add_pool adds no trainable parameters

        - It performs a fixed sum according to graph membership

- Inspected the representation shapes through the complete model

    - The input had shape [585, 7]

        - The minibatch contained 585 nodes across 32 MUTAG graphs

        - Each node started with seven categorical input features

    - After conv1, the shape was [585, 64]

        - There was still one row per node, but each node now had a 64-dimensional learned representation

    - After the first ReLU, the shape remained [585, 64]

    - After conv2, the shape remained [585, 64]

        - The second message-passing layer retained one 64-dimensional representation for each node

    - After the second ReLU, the shape again remained [585, 64]

    - After global sum pooling, the shape changed to [32, 64]

        - The first dimension changed from 585 node rows to 32 graph rows

        - The second dimension remained 64 because summing node vectors does not change their feature width

        - Each graph was represented by one 64-dimensional vector

        - The observed shape was consistent with graph-level pooling; the shape alone did not verify the numerical grouping of every node

    - After the classifier, the shape changed to [32, 2]

        - There was one row for each graph

        - The 64 graph features were transformed into two class logits

    - Calling the complete model also returned shape [32, 2]

        - This established successful forward execution with the expected graph-classification output dimensions

        - No numerical comparison between the manual sequence and complete forward output was performed

- Added a small two-graph example to demonstrate global_add_pool independently of the GCN

    - torch.ones((5, 2)) created five node representations

        - Each node had the two-dimensional representation [1, 1]

    - toy_batch was [0, 0, 0, 1, 1]

        - The first three node representations were assigned to graph 0

        - The final two node representations were assigned to graph 1

    - global_add_pool summed the node representations separately for the two graph IDs

    - Graph 0 contained three copies of [1, 1]

        - Its expected pooled representation was [3, 3]

    - Graph 1 contained two copies of [1, 1]

        - Its expected pooled representation was [2, 2]

    - The actual pooled output was:

        - [3, 3] for graph 0

        - [2, 2] for graph 1

    - torch.equal compared the actual tensor with the manually specified expected tensor

        - Toy pooling correct was True

        - This established on a simple known example that global_add_pool grouped nodes by their batch IDs and summed them as expected

- Execution

    - Ran python -m experiments.models.inspect_gcn successfully

    - The complete GCN contained 4,802 trainable parameters

    - Sum pooling changed the real minibatch representation from [585, 64] node embeddings to [32, 64] graph embeddings

    - The classifier changed the graph embeddings from [32, 64] to [32, 2] graph-class logits

    - The complete forward method returned shape [32, 2]

    - The independent two-graph pooling example produced the expected sums and returned Toy pooling correct: True

- Aligned the current model and inspector with the later code-consistency decisions

    - The GCN constructor now uses num_features, hidden_dim and num_classes

        - num_features replaces the earlier in_channels name

        - hidden_dim replaces the earlier baseline_width constructor argument

        - The renamed arguments retain the same numerical roles and model dimensions

    - The model uses import torch.nn as nn and import torch.nn.functional as F, matching the other model implementations

    - The class docstring was removed

    - The inspector uses model.eval() and one torch.no_grad() context around the manual and complete forward calculations

        - model.eval() selects evaluation behaviour

        - torch.no_grad() disables gradient recording

    - The shape labels now consistently identify input node features, each convolution, each ReLU, sum pooling, classification and the complete model output shape

    - The existing pooling toy remains in the GCN inspector

    - These source changes preserve the architecture and parameter counts; the execution observations above remain the recorded Stage 2 observations

- 2.2 added sum pooling and graph classifier





# 2.3 Ordinary Fitting and Validation

- Added experiments/train.py to introduce ordinary supervised fitting and validation

    - The complete GCN from 2.1 and 2.2 could now be fitted to labelled MUTAG graphs rather than only inspected with untrained parameters

    - Training uses the development training partition to update model parameters

    - Validation measures the current model after every training epoch without directly updating its parameters

    - The development-test partition remained unused in the final 2.3 implementation

        - Test assessment would be introduced after validation-based model-state selection and restoration in 2.4

- Kept the training implementation simple while making the shared operations suitable for later GNN models

    - train_epoch receives the model as an argument rather than containing GCN-specific layer operations

    - evaluate also receives the model as an argument

    - Both functions rely on the common graph-classification interface

        - The model receives x, edge_index and batch

        - The model returns one row of graph-class logits for every graph in the minibatch

    - GraphSAGE, GIN, GAT and GATv2 are intended to expose the same interface

    - The functions remained in experiments/train.py at this stage because only one model used them

        - They were subsequently moved into shared source modules in Stage 3

- Added a settings dictionary containing the choices that defined the development fit

    - dataset=MUTAG selected the development dataset

    - baseline_width=64 recorded the fixed GCN representation width used by this run

        - The current runner calls this setting hidden_dim

        - Historical settings and results retain the original baseline_width field

    - learning_rate=0.01 gave the Adam learning rate

        - The learning rate influences the size of parameter updates made during optimisation

    - weight_decay=0.0005 gave the fixed Adam weight-decay setting

        - Weight decay discourages excessively large parameter values during optimisation

    - epochs=1000 gave the fixed number of complete training epochs

        - The setting is called epochs rather than max_epochs because the procedure deliberately completes all 1,000 epochs

        - No patience or early-stopping rule is used

    - batch_size=32 requested up to 32 graphs in each minibatch

    - seed=0 controlled stochastic behaviour during model fitting

    - split_seed=0 reproduced the fixed development partition established in Stage 1

    - Ordinary unchanged Adam defaults remained library defaults rather than being added as experimental choices

- Organised the original experiment runner around its execution sequence

    - The Stage 2 runner placed this sequence inside main

        - It loaded MUTAG

        - It obtained the development partition indices

        - It set the training seed

        - It created the DataLoaders

        - It selected the compute device

        - It constructed the GCN

        - It constructed the Adam optimiser

        - It ran the training and validation epochs

    - The original __name__ condition called main when experiments.train was executed as a module

        - This prevented a training run from beginning merely because that version of the module was imported

    - A main function is not required for python -m execution

        - The same experiment sequence can be written directly at module level

        - A sequential top-level runner executes that sequence when imported as well as when run directly

        - This organisational choice does not change the training calculations performed by the normal execution command

- Reused load_dataset to load MUTAG consistently with Stage 1

    - The model used the same seven categorical node-feature channels previously inspected

    - edge_index supplied graph connectivity

    - Edge features remained excluded from the principal model input

- Reused stratified_split to reconstruct the fixed development partitions

    - train_indices identified the graphs used for parameter updates

    - val_indices identified the graphs used for validation

    - The third returned partition was assigned to _ because development-test evaluation did not belong in 2.3

    - split_seed and seed have different purposes

        - split_seed determines which original graphs belong to each development partition

        - seed controls stochastic behaviour during fitting without changing graph membership

- Seeded the stochastic fitting process

    - set_seed seeds the Python, NumPy and PyTorch random-number generators

    - A separate torch.Generator was created for the shuffled training DataLoader

    - manual_seed gave this generator the same training seed

    - Supplying the generator to the DataLoader controlled the shuffled training order

    - Keeping loader shuffling explicitly seeded prevented its random sequence from depending on how much randomness a particular model consumed during construction

        - This is useful when several GNN models share the same fitting machinery

    - Seeding does not by itself guarantee bit-for-bit identical execution across devices or environments

- Created separate training and validation DataLoaders

    - The training loader received dataset[train_indices]

    - batch_size=32 requested up to 32 graphs per training minibatch

    - shuffle=True changed the graph ordering used to form training minibatches across epochs

    - The validation loader received dataset[val_indices]

    - shuffle=False kept validation ordering fixed

- Confirmed the expected development partition sizes during execution

    - Training graphs: 150

    - Validation graphs: 18

    - These matched the Stage 1 development split

- Confirmed the expected minibatch counts

    - Training batches: five

        - Four full training minibatches contained 32 graphs each

        - The remaining 22 graphs formed the fifth minibatch

        - 32 + 32 + 32 + 32 + 22 = 150

    - Validation batches: one

        - All 18 validation graphs fitted into one minibatch because the requested batch size was 32

- Added automatic device selection

    - torch.cuda.is_available() checks whether CUDA is available to PyTorch

    - torch.device selects CUDA when available and otherwise selects CPU

    - model.to(device) moves the model parameters and buffers to the selected device

    - graph_batch.to(device) moves the tensors in each PyG minibatch to the same device before model computation

    - Execution reported Device: cuda

        - The development fit ran using the available CUDA device

- Created the Adam optimiser

    - model.parameters() supplies the registered model parameters

        - The optimiser receives those parameters through the same interface for every model

        - All 4,802 parameters in this GCN were trainable

    - Adam uses gradients of the loss to update parameter values

    - learning_rate=0.01 and weight_decay=0.0005 were supplied explicitly

    - The remaining unchanged Adam behaviour used the library defaults

- Added train_epoch for one complete optimisation pass through the training partition

    - One epoch processes each of the 150 training graphs once

    - model.train() places the model into training mode before minibatches are processed

        - The current GCN does not contain dropout or batch-normalisation layers whose behaviour changes between training and evaluation modes

        - Using the appropriate mode keeps the function suitable for models where that distinction matters

    - Every graph_batch is moved to the selected device before model operations are performed

- Implemented the minibatch optimisation sequence explicitly

    - optimizer.zero_grad() clears parameter gradients remaining from the previous minibatch

        - PyTorch accumulates gradients by default

        - Clearing them ensures that the following update uses gradients from the current minibatch rather than unintentionally combining them with previous gradients

    - Calling the model performs the forward pass

        - graph_batch.x supplies node features

        - graph_batch.edge_index supplies graph connectivity

        - graph_batch.batch identifies graph membership for sum pooling

        - The GCN produces one row of raw class logits for every graph in the minibatch

    - F.cross_entropy compares the logits with graph_batch.y

        - graph_batch.y contains the correct class index for each graph

        - Cross-entropy accepts raw logits directly, so the model does not apply softmax before calculating the loss

        - With the selected usage, the returned value is the mean classification loss across the graphs in the minibatch

    - loss.backward() performs backpropagation

        - PyTorch traces backwards through the operations that produced the loss

        - It calculates gradients describing how the loss locally changes with the trainable parameters

        - backward calculates gradients but does not itself update the parameter values

    - optimizer.step() performs the Adam parameter update

        - Adam uses the calculated gradients and its optimiser state to change parameter values

    - This zero-gradient, forward, loss, backward and update sequence occurs for each of the five training minibatches in every epoch

- Added graph-level predictions and accuracy

    - logits.argmax(dim=1) selects the index of the largest class logit for each graph

    - dim=1 is the class dimension of the graph-by-class output tensor

    - The predicted class indices are compared with graph_batch.y using ordinary ==

    - Summing the comparison results counts correct predictions

    - .item() converts the scalar count into a Python number for accumulation

    - Accuracy is the total number of correct predictions divided by the total number of graphs

- Calculated partition loss as a mean over graphs rather than an unweighted mean of minibatch means

    - F.cross_entropy returns the mean loss within the current minibatch

    - The final training minibatch contains 22 graphs while the preceding minibatches contain 32

    - Giving all five minibatch means equal weight would give each graph in the smaller final minibatch more influence

    - graph_batch.num_graphs gives the number of graphs in the current minibatch

    - loss.item() extracts the scalar minibatch loss used for metric reporting

    - Multiplying the minibatch mean by num_graphs gives its contribution to the total graph loss

    - These contributions are accumulated in total_loss

    - Dividing by total_graphs produces the mean cross-entropy per graph

    - Accuracy follows the same graph-level principle by accumulating correct predictions and total graph counts

- Distinguished training metrics from a separate evaluation of the training partition

    - Each minibatch's logits are calculated before that minibatch's optimiser update

    - The same logits supply its loss and predicted classes for reporting

    - Model parameters change between minibatches as optimisation proceeds

    - The reported training loss and accuracy therefore combine observations made at successive parameter states during the epoch

    - They are not a separate evaluation of the final epoch state on all training graphs

- Added evaluate for validation

    - evaluate measures loss and accuracy without updating model parameters

    - model.eval() places the model into evaluation mode

    - torch.no_grad() disables gradient recording during validation calculations

        - Validation performs no backpropagation

        - Disabling gradient recording avoids retaining an unnecessary computation graph

    - Validation uses the same model forward interface, cross-entropy definition and prediction rule as training

        - The metric definitions match

        - Validation evaluates the fixed state reached after the completed training epoch

    - Validation does not call optimizer.zero_grad(), loss.backward() or optimizer.step()

        - Validation graphs measure the current fitted model without directly updating its parameters

- Added the fixed 1,000-epoch training loop

    - Epoch numbers run from one through 1,000

    - train_epoch first performs a complete optimisation pass through the training partition

    - evaluate then measures the resulting model on the validation partition

    - Training loss, training accuracy, validation loss and validation accuracy are printed every ten epochs

    - Validation is calculated after every epoch even though only every tenth set of metrics is printed

        - This distinction matters in 2.4 when validation loss selects the model state

    - No patience or early stopping is used

        - Every fit completes the same fixed number of epochs

        - Stage 2.4 selects the lowest-validation-loss state from those epochs without using validation to decide when fitting stops

- Ran python -m experiments.train successfully for all 1,000 epochs

    - The final 2.3 run completed without a loading, device, forward-pass, loss, backpropagation, optimiser or validation error

    - Its printed settings were:

        - dataset: MUTAG

        - baseline_width: 64

        - learning_rate: 0.01

        - weight_decay: 0.0005

        - epochs: 1000

        - batch_size: 32

        - seed: 0

        - split_seed: 0

- Training loss generally decreased during the final 2.3 run

    - At epoch 10, training loss was 0.5014

    - At epoch 100, training loss was 0.3849

    - At epoch 500, training loss was 0.3200

    - At epoch 1,000, training loss was 0.2928

    - The decrease was not monotonic

        - Training loss rose to 0.3969 at epoch 210 and to 0.3566 at epoch 480 before falling again

        - Such fluctuations are compatible with minibatch optimisation on shuffled data

- Training accuracy generally increased compared with the beginning of the run

    - At epoch 10, training accuracy was 0.6867

        - The rounded value corresponds to 103 correct predictions out of 150

    - At epoch 100, training accuracy was 0.8400

    - At epoch 500, training accuracy was 0.8733

    - At epoch 1,000, training accuracy was 0.8667

    - The highest training accuracy among the printed ten-epoch checkpoints was 0.9067 at epoch 740

    - Training accuracy fluctuated while parameter updates continued throughout fitting

- Validation loss initially improved substantially but later became more variable

    - Validation loss was 0.4496 at epoch 10

    - It had fallen to 0.3296 by epoch 100

    - It was 0.2772 at epoch 200

    - Among the printed ten-epoch checkpoints, the lowest visible validation loss was 0.2633 at epoch 250

    - The printed checkpoints do not establish that epoch 250 was the actual minimum-validation-loss epoch

        - Validation was calculated after every epoch but only every tenth result was printed

        - An unprinted epoch may have produced a lower validation loss

        - Stage 2.4 adds explicit tracking of the minimum across every epoch

- Validation loss later increased while training loss generally remained lower

    - Validation loss was 0.3226 at epoch 540

    - It was 0.3562 at epoch 680

    - It reached 0.3765 at epoch 830

    - It reached 0.3949 at epoch 960

    - It was 0.3354 at epoch 1,000

    - The epoch-1,000 validation loss was higher than the best values visible earlier in training

    - The final parameter state should therefore not automatically be treated as the state preferred by the minimum-validation-loss criterion

    - Stage 2.4 retains and restores the lowest-validation-loss state across the fixed training budget

- Validation accuracy was highly discrete because the partition contained only 18 graphs

    - Changing one graph from incorrectly to correctly classified changes accuracy by 1 / 18, approximately 0.0556

    - Frequently observed validation accuracies included:

        - 0.8333, corresponding to 15 correct predictions out of 18

        - 0.8889, corresponding to 16 correct predictions out of 18

        - 0.9444, corresponding to 17 correct predictions out of 18

    - Validation accuracy sometimes changed while validation loss moved in a different direction

        - Accuracy records whether the highest-scoring class matches the target

        - Cross-entropy also reflects the probability assigned to the target class

    - The project uses minimum validation cross-entropy to select model states

- Retained the distinct observations from the earlier 2.3 development run

    - That run also completed 1,000 epochs

    - Its training losses at epochs 10, 100, 500 and 1,000 were 0.5014, 0.3861, 0.3174 and 0.2938 respectively

    - Its training accuracies at those epochs were 0.6867, 0.8533, 0.8867 and 0.8667 respectively

    - Its highest printed training accuracy was 0.9067 at epoch 740

    - Its validation losses at epochs 10, 100 and 200 were 0.4496, 0.3294 and 0.2772 respectively

    - Its lowest validation loss among the printed checkpoints was 0.2594 at epoch 250

    - Later printed validation losses included 0.3407 at epoch 560, 0.3789 at epoch 680, 0.3730 at epoch 830 and 0.3867 at epoch 890

    - These values belong to the earlier run and are not combined with the final 2.3 run as though they describe one trajectory

    - The earlier references to introducing patience or early stopping were superseded by the fixed-epoch procedure

- The final 2.3 implementation did not return or report a selected epoch

    - Its purpose was to establish ordinary fitting and validation first

    - The final in-memory model was the state produced after epoch 1,000

    - No claim was made that this final state minimised validation loss

    - Stage 2.4 would add tracking of:

        - The epoch with the lowest validation loss

        - That epoch's validation loss

        - That epoch's validation accuracy

        - An independent copy of that epoch's model state

    - After all 1,000 epochs, the selected state would be restored before development-test assessment

- The 2.3 run established that the supervised GCN fitting path worked end to end

    - The fixed development split was reconstructed

    - Training minibatches were processed using the seeded shuffle generator

    - The model and graph tensors ran on CUDA

    - Raw graph logits were compared with graph labels using cross-entropy

    - Backpropagation calculated parameter gradients

    - Adam updated the GCN and classifier parameters

    - Training loss and accuracy were accumulated over individual graphs

    - Validation loss and accuracy were measured without parameter updates

    - All 1,000 fixed epochs completed

    - Model-state selection and restoration remained deferred to 2.4

- 2.3 added ordinary fitting and validation





# 2.4 Validation-Selected State and Restoration

- Extended the fixed 1,000-epoch fitting procedure with validation-based model-state selection

    - Ordinary fitting in 2.3 evaluated validation after every epoch without preserving a particular state

    - The model remaining after 2.3 was the state produced by epoch 1,000

    - Earlier epochs could have lower validation loss than the final epoch

    - Stage 2.4 selects the state with the lowest validation loss across the complete fit

- Kept the fixed training length of 1,000 epochs

    - No patience or early stopping is used

    - Every fit completes exactly the number of epochs specified by settings["epochs"]

    - Validation determines which fitted state is retained, but does not determine when fitting stops

    - This keeps the epoch budget fixed while allowing selection of an earlier state

- Added train_model to contain fixed-epoch fitting and validation-state selection

    - train_model receives the model, training loader, validation loader, optimiser, device and number of epochs

    - It reuses train_epoch for parameter optimisation

    - It reuses evaluate for validation measurement

    - It does not depend on GCN-specific layers

        - It requires a graph classifier that accepts x, edge_index and batch and returns graph-class logits

        - The same fitting procedure can be reused by GraphSAGE, GIN, GAT and GATv2

    - The function remained in experiments/train.py at this substage and was subsequently moved into src/training.py in Stage 3

- Added best_epoch to record the selected epoch

    - best_epoch starts at zero before any state has been evaluated

    - Validation loss is inspected after every completed training epoch

    - When validation loss becomes strictly lower than the previous minimum, best_epoch is replaced with the current epoch number

    - Selection is performed every epoch even though progress is printed every ten epochs

        - The selected epoch does not need to appear in the printed progress output

- Added best_val_loss as the model-selection criterion

    - best_val_loss starts at positive infinity using float("inf")

    - The first finite validation loss therefore becomes the initial selected value

    - A later epoch replaces it only when val_loss < best_val_loss

    - An exactly equal validation loss does not replace the existing selected state

        - The earlier epoch is retained in an exact tie

    - Validation cross-entropy determines which state is selected

- Added best_val_accuracy to record the accuracy belonging to the selected state

    - best_val_accuracy is updated only when validation loss strictly improves

    - It records the validation accuracy produced by the same epoch as best_val_loss

    - Validation accuracy does not independently choose another epoch

    - best_epoch, best_val_loss and best_val_accuracy describe one consistent model state

- Preserved the selected model state using PyTorch tensor cloning

    - model.state_dict() provides the model's named parameters and persistent buffers

    - Training continues after a state becomes selected, so the live model parameters continue to change

    - Recording only the selected epoch number would not preserve its parameter values

    - When validation loss strictly improves, every tensor in model.state_dict() is copied using value.clone()

    - clone creates a tensor containing the same numerical values with independent storage

        - Later optimiser updates do not alter the cloned values stored in best_state

    - A dictionary comprehension retains the original state-dictionary names while cloning each tensor

    - Only one selected state is retained at a time

        - If a later epoch improves validation loss, its cloned state replaces best_state

        - A separate model is not stored for every epoch

    - Using clone keeps state preservation within PyTorch without introducing copy or deepcopy

- Restored the validation-selected state after all 1,000 epochs

    - The training loop always continues through the complete epoch budget

    - At the end of the loop, the live model initially contains the state produced by the final epoch

    - model.load_state_dict(best_state) restores the preserved parameters and buffers from the selected epoch

    - The model used after train_model therefore corresponds to the minimum-validation-loss selection

- Added selected-state statistics as return values from train_model

    - best_epoch gives the selected epoch

    - best_val_loss gives that epoch's graph-mean validation cross-entropy

    - best_val_accuracy gives the accuracy produced by the same state

    - At this substage, train_model returned those three values after restoration

    - Stage 2.5 subsequently added training_seconds as a fourth return value

    - The model itself does not need to be returned separately because load_state_dict updates the existing model object

- Kept restoration handling minimal

    - No duplicate validation evaluation is performed immediately after restoration

    - The selected validation loss and accuracy were already measured when that state was encountered

    - The implementation selects, preserves and restores the state before subsequent assessment

- Added the development-test DataLoader

    - test_indices was retained from stratified_split rather than discarding the third partition

    - dataset[test_indices] selected the 20 development-test graphs established in Stage 1

    - batch_size remained 32

    - shuffle=False kept the evaluation order fixed

    - All 20 graphs fitted into one minibatch

- Added one preliminary development-test assessment after model selection

    - The test loader is not supplied to train_model

        - Its graphs are not used by the fitting function for gradients, validation measurement or epoch selection

    - Test evaluation occurs after all 1,000 epochs and restoration of the selected state

    - The existing evaluate function is reused

        - model.eval() selects evaluation behaviour

        - torch.no_grad() disables gradient recording

        - No parameter updates occur

    - The output was labelled Preliminary development test

        - It provides an initial assessment of the development procedure

        - It is not the final project performance estimate

- Ran python -m experiments.train successfully for the complete fit

    - The run used:

        - dataset: MUTAG

        - baseline_width: 64

        - learning_rate: 0.01

        - weight_decay: 0.0005

        - epochs: 1000

        - batch_size: 32

        - seed: 0

        - split_seed: 0

    - Execution used CUDA

    - The partition contained:

        - 150 training graphs

        - 18 validation graphs

        - 20 development-test graphs

- Training continued through epoch 1,000

    - There was no patience counter or early-stopping condition

    - The final printed epoch had:

        - Training loss: 0.2937

        - Training accuracy: 0.8800

        - Validation loss: 0.3095

        - Validation accuracy: 0.9444

    - The final epoch did not have the lowest validation loss encountered during fitting

- The selected model occurred at epoch 337

    - Selected epoch: 337

    - Selected validation loss: 0.2589

    - Selected validation accuracy: 0.9444

    - The rounded validation accuracy corresponds to 17 correct predictions out of 18

    - Epoch 337 was absent from the ten-epoch progress output

        - This demonstrates that selection was not restricted to the printed checkpoints

        - The code establishes that validation and selection are performed after every epoch

        - Restricting selection to multiples of ten would have missed this selected state

- Compared the selected and final validation losses

    - Selected validation loss at epoch 337: 0.2589

    - Validation loss at epoch 1,000: 0.3095

    - The selected state had lower validation loss than the final state

    - Restoring epoch 337 ensured that development-test evaluation used the state chosen by the fixed selection criterion

- Distinguished the selected loss from the selected accuracy

    - Validation accuracy of 0.9444 appeared at other epochs during training

    - Epoch 337 was selected with validation loss 0.2589

    - The stored summary does not establish that no later epoch had an exactly equal minimum loss

        - The strict comparison retains the earliest exact tie

        - Rounded printed losses do not establish exact equality between the underlying values

    - Validation loss determines the selected state

    - Validation accuracy describes that same state after it has been selected

- Evaluated the restored model once on the development-test partition

    - Preliminary development-test loss: 0.5301

    - Preliminary development-test accuracy: 0.7500

    - This corresponds to 15 correctly classified graphs and five incorrectly classified graphs

    - The development-test accuracy was lower than the selected validation accuracy of 0.9444

        - Validation contained 18 graphs and development test contained 20 graphs

        - Validation was also used for epoch selection

        - The observed difference did not justify changing the fixed training or selection procedure

    - Test assessment occurred after validation-based selection and did not determine the selected epoch

- Distinguished the roles of fitting, selection and assessment

    - Training data determines parameter updates

    - Validation loss selects one state from the states reached during the fixed training budget

    - clone preserves that state independently of subsequent optimisation

    - load_state_dict restores the selected state after fitting

    - The selected epoch and validation statistics are returned explicitly

    - The development test provides one preliminary assessment of the restored state

- Preserved this fit as separate evidence from the later recorded fit

    - Stage 2.4 selected epoch 337 and obtained development-test accuracy 0.7500

    - Stage 2.5 conducted a separate fit with result recording

    - Its different selected epoch and metrics do not replace the observations from this run

- 2.4 added validation state selection and restoration





# 2.5 Result Recording and Source Provenance

- Added src/recording.py to preserve development experiment evidence as JSON

    - The recorder remains a small collection of ordinary functions

    - get_source_commit runs git rev-parse HEAD through subprocess to obtain the current Git commit identifier

        - subprocess is needed because the identifier comes from Git

        - The identifier does not describe uncommitted source changes

        - Experiment source and effective settings must therefore be committed before a recorded fit

        - The entire repository does not need to be clean if unrelated changes are present

    - get_result_path constructs a descriptive path from the result purpose, model, dataset, optional variant and training seed

        - The GCN development result is stored as results/development_gcn_mutag_seed0.json

    - prepare_result_path creates the results directory when it does not already exist

        - os.makedirs with exist_ok=True creates the directory when needed and leaves an existing directory unchanged

        - The function checks whether the intended result file already exists before training starts

        - This avoids beginning a fit whose intended result filename is already occupied

    - save_result writes JSON using file mode x

        - Mode x refuses to replace an existing file

        - The early destination check and the write-time protection serve separate purposes

    - save_result records the PyTorch, PyG and CUDA versions automatically

- Added model and variant identity to the effective settings

    - model=GCN identifies the model in the result and its filename

    - variant=None records that this GCN has no special experimental variant

        - Python None becomes null when written as JSON

    - The recorded settings were:

        - dataset=MUTAG

        - baseline_width=64

        - learning_rate=0.01

        - weight_decay=0.0005

        - epochs=1000

        - batch_size=32

        - seed=0

        - split_seed=0

    - The existing result retains baseline_width because that was the setting name used for this fit

        - Current source uses hidden_dim for the same width

        - Renaming the source setting does not require rewriting the historical JSON

    - Ordinary unchanged Adam defaults remain library defaults rather than additional experimental factors

- Added source provenance before the recorded fit

    - The successful fit records source commit 9898a4a8d4f8da9baf5460dcc28ee95d7422eaba

    - This identifies the committed experiment source and settings used for the fit

    - The result JSON and completed log are committed afterwards

    - The result commit is therefore distinguished from the source commit that produced the fit

- Added training-pass runtime measurement

    - time.perf_counter() is used as the wall-clock timer

    - The timed operation is the complete train_epoch call

    - It includes:

        - Iterating through the training DataLoader

        - Moving graph minibatches to the selected device

        - Forward computation

        - Cross-entropy calculation

        - Backpropagation

        - Adam parameter updates

        - Class prediction and correct-prediction counting

        - Loss and accuracy accumulation and the final metric calculations

    - Validation, checkpoint copying and restoration, progress printing, development-test evaluation and result writing remain outside the timed training call

    - CUDA work can execute asynchronously relative to Python

        - torch.cuda.synchronize() is called before starting the timer and after the training call when CUDA is used

        - The ending synchronisation completes before elapsed time is calculated

        - This makes the measured interval include completion of the training call's GPU work

    - training_seconds accumulates the measured intervals across all epochs

    - train_model returns training_seconds alongside the selected epoch, validation loss and validation accuracy

    - Dividing training_seconds by the completed epoch count gives mean training-pass time per epoch

        - This is not the complete elapsed duration of the experiment command

- Kept the validation-selected state procedure from 2.4

    - Every fit completes the fixed 1,000-epoch budget

    - No patience or early stopping is used

    - Validation is evaluated after every epoch

    - Strictly lower validation cross-entropy replaces the selected state

    - An exact validation-loss tie retains the earlier epoch

    - The selected tensors are preserved with clone and restored after the final epoch

- Added direct parameter counting to the recorded evidence

    - parameter.numel() gives the number of scalar values in a parameter tensor

    - Summing numel across model.parameters() gives the total parameter count

    - Filtering parameters by requires_grad gives the trainable parameter count

    - Buffers are excluded from both counts because they are not registered parameters

- Added the dataset loader and feature policy to the result

    - cleaned=False records use of the ordinary TU dataset rather than its cleaned variant

    - use_node_attr=False records that additional continuous node attributes are not requested

    - use_edge_attr=False records that additional continuous edge attributes are not requested

    - These flags do not remove all categorical labels

    - MUTAG categorical node labels are used as the node features supplied to the GCN

    - Connectivity is used through edge_index

    - Edge features are not supplied to the GCN forward pass

- Added the original development partition indices

    - The JSON contains the original dataset indices for all three partitions

    - Training contains 150 graphs

    - Validation contains 18 graphs

    - Development test contains 20 graphs

    - These counts account for all 188 MUTAG graphs

    - Recording indices preserves the identities of the graphs used in each role rather than only the partition sizes

- Added validation-selection evidence to the JSON

    - The criterion is recorded as minimum validation cross-entropy

    - The tie rule is recorded as earliest exact tie

    - early_stopping=False records the fixed-epoch procedure

    - All 1,000 epochs completed

    - Epoch 384 was selected

        - Validation loss: 0.256849080324173

        - Validation accuracy: 0.8333333333333334

        - This accuracy corresponds to 15 correct predictions out of 18

    - Some printed epochs reached higher validation accuracy

        - They were not selected because loss determines the selected state

    - Epoch 384 was absent from the ten-epoch reporting points

        - This demonstrates selection beyond the printed checkpoints

        - The training code establishes that selection is evaluated after every epoch

- Added the development-test result

    - Assessment occurred after restoration of the selected state

    - Development-test loss: 0.49510836601257324

    - Development-test accuracy: 0.8

        - This corresponds to 16 correct predictions out of 20

    - The result remains development evidence rather than final cross-validation evidence

- Recorded model size and runtime

    - Total parameters: 4,802

    - Trainable parameters: 4,802

        - This agrees with the complete GCN parameter count established in 2.2

    - The 1,000 timed training passes took 37.63074249937199 seconds in total

    - Mean training-pass duration was 0.03763074249937199 seconds per epoch

        - This is approximately 37.6 milliseconds per epoch

- Recorded the execution environment

    - The run used CUDA on an NVIDIA GeForce RTX 4070 Laptop GPU

    - PyTorch version: 2.13.0+cu130

    - PyG version: 2.8.0.post1

    - CUDA version: 13.0

    - These values provide the device and software context for interpreting execution and runtime

- Inspected the development JSON

    - results/development_gcn_mutag_seed0.json contains the purpose, effective settings, loader policy, feature policy, partition indices, selection details, development-test metrics, parameter counts, runtime information, device information, software versions and source commit

    - Its values agree with the successful terminal execution

    - The source commit matches the identifier printed before training began

- A recording failure exposed a missing filesystem requirement

    - An earlier 1,000-epoch attempt completed training but failed when saving because the results directory did not exist

    - prepare_result_path was changed to create that directory before fitting

    - The successful recorded fit subsequently completed and saved the JSON

    - The unsuccessful saving attempt and the successful recorded fit are separate executions

- Aligned the recording source with the later code-consistency decisions

    - Removed the recording-function docstrings

    - Renamed the local filename variable in get_result_path to result_path because it includes both a directory and filename

    - Formatted json.dump across multiple lines to match the other save call

    - These changes preserve path construction, result contents and overwrite protection

    - The runtime wording in the current runner now explicitly includes metric calculation and accumulation

        - This clarifies the existing timing scope rather than changing the measurements already recorded

- 2.5 recorded the GCN development fit





# 2 Closing Notes

- Decisions

    - GCN is the first contextual graph-classification baseline

        - It uses two GCNConv layers

        - MUTAG node features enter with width seven

        - Both graph-convolution layers produce width 64

        - ReLU follows both convolutions

        - Global sum pooling produces one 64-dimensional representation per graph

        - A linear classifier maps that representation to two MUTAG class logits

    - GCNConv retains the fixed operator settings established in Stage 2

        - improved=False

        - cached=False

        - add_self_loops=True

        - normalize=True

        - bias=True

        - Connectivity normalisation must not be cached across minibatches containing different graphs

    - Principal model inputs remain categorical node features and graph connectivity

        - MUTAG node labels supply x

        - edge_index supplies connectivity

        - Available edge features are excluded from the forward pass

    - Global sum pooling remains the shared graph readout

        - This is the supervisor-recommended readout

        - Its mathematical result is invariant to node ordering within each graph

    - Development training retains the fixed shared settings

        - Adam optimiser

        - Learning rate 0.01

        - Weight decay 0.0005

        - Batch size 32

        - Training seed zero

        - Development split seed zero

        - Exactly 1,000 epochs

        - No patience or early stopping

    - Model-state selection uses minimum validation cross-entropy

        - Validation is evaluated after every epoch

        - Strictly lower loss replaces the selected state

        - An exact tie retains the earlier epoch

        - Selected parameters and persistent buffers are cloned

        - The selected state is restored after the complete training budget

    - Development-test assessment occurs after selection and restoration

        - Its results are preliminary development evidence

        - They must not determine model settings or the selected epoch

        - Final research performance comes from the later locked cross-validation protocol

    - Recorded fits preserve source and execution provenance

        - Experiment source and effective settings are committed before fitting

        - Results record the source commit that produced them

        - Existing result files are not silently overwritten

        - Original partition indices, environment details and the timing convention are retained

    - Runtime measures the training pass

        - It includes data loading, device transfer, model and loss computation, backward calculation, optimiser updates and metric calculation and accumulation

        - It excludes validation, checkpoint copying and restoration, progress printing, development-test assessment and result writing

    - Current source naming and formatting follow the shared code conventions

        - The GCN constructor uses num_features, hidden_dim and num_classes

        - Model and utility functions omit docstrings and type annotations

        - Models use the agreed import style and a direct forward sequence

        - Historical result files retain the settings names and values used when their fits ran

    - Model inspectors primarily expose shapes and parameter structure

        - Use the same common coverage, variable names, print labels and execution conventions across models

        - Retain model.eval() and torch.no_grad() for forward inspection

        - Retain the small GCN pooling toy where pooling was introduced

        - Additional mechanism explanations remain secondary and should not require reconstructing the convolution implementations

- Ideas

    - Seeded CUDA training does not necessarily imply bit-for-bit identical numerical execution

        - Investigate strict determinism only if it becomes a concrete reproducibility requirement

    - Runtime comparisons can provide useful practical context when models use the same timing convention and comparable execution conditions

- Report notes

    - The project GCN adapts the Kipf and Welling operator to graph classification

        - It does not reproduce the paper's complete transductive node-classification architecture

    - The complete MUTAG classifier contains 4,802 parameters, all trainable

    - Stage 2.4 and Stage 2.5 contain separate development fits

        - Stage 2.4 selected epoch 337 and obtained development-test accuracy 0.7500

        - Stage 2.5 selected epoch 384 and produced the preserved JSON result

    - The recorded Stage 2.5 fit completed all 1,000 epochs

        - Selected validation loss: 0.256849080324173

        - Selected validation accuracy: 0.8333333333333334

        - Development-test loss: 0.49510836601257324

        - Development-test accuracy: 0.8

    - Its timed training passes took 37.63074249937199 seconds in total

        - Mean training-pass duration was approximately 0.0376 seconds per epoch

    - The recorded fit used CUDA on an NVIDIA GeForce RTX 4070 Laptop GPU

    - The result is stored in results/development_gcn_mutag_seed0.json

    - Its source commit is 9898a4a8d4f8da9baf5460dcc28ee95d7422eaba

    - Later naming and formatting corrections do not replace the historical result or establish a new training run

    - These development measurements should not be compared directly with the original GCN paper's node-classification results because the tasks, datasets, architecture and evaluation protocols differ