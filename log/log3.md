# 3.1 Shared Training and Evaluation Functions

- Extracted the existing training and evaluation functions so GraphSAGE and GIN can reuse the GCN training procedure

    - Each model accepts node features, connectivity and graph membership through model(x, edge_index, batch), and returns graph-class logits

    - The training code therefore works with the supplied model without needing to know which graph convolution it uses

- Moved train_epoch and train_model into src/training.py

    - train_epoch performs one pass through the training loader and returns loss and accuracy

    - train_model calls train_epoch and evaluate each epoch, selects the lowest-validation-loss state and restores it after fitting

    - These functions receive their inputs through arguments rather than reading the development runner's settings directly

- Moved evaluate into src/evaluation.py

    - Both validation and development-test assessment need the same loss and accuracy calculation without parameter updates

    - src/training.py imports evaluate for validation, while experiments/train.py imports it for development-test assessment

- Updated experiments/train.py to import the extracted functions and removed their original definitions

    - The runner still constructs the experiment: settings, data loaders, model and optimiser, followed by fitting, assessment and result recording

    - Passing model into train_model gives the function access to the same model object used by the runner

    - Restoring the selected parameters inside train_model therefore updates the model subsequently assessed by the runner

    - Function bodies, arguments and returned values were preserved, so the extraction changes code organisation without changing the intended training procedure

- Chose python -m src.training as the module-loading check

    - This loads src.training and its src.evaluation import

    - Function definitions are loaded without calling them, so no training or printed results are expected

    - Successful module loading checks the imports, but does not establish that a training run produces equivalent results

- Commit: 3.1 extracted shared training and evaluation functions





# 3.2 GraphSAGE

- Added src/models/graphsage.py to implement a GraphSAGE graph classifier using PyG's SAGEConv operator

    - SAGEConv is a message-passing layer that creates an updated representation for every node by combining information from that node and its neighbours

    - With the chosen settings, it first calculates the elementwise mean of the incoming neighbour representations

    - It applies one learned linear transformation to this mean and a separate learned linear transformation to the receiving node's own representation, then adds the results and a learned bias

    - The operation is W_neighbour × mean(neighbour vectors) + W_self × node vector + bias; the model applies ReLU afterwards

    - For example, neighbour vectors [1, 3] and [3, 1] have mean [2, 2]; the layer transforms this aggregate separately from the receiving node's own vector

    - The mean has no learned parameters, but the two transformations learn how to combine feature coordinates and how self and neighbour information contribute to the output

- Constructed the model with num_features, hidden_dim and num_classes

    - num_features is the number of input features per node, supplied by dataset.num_node_features; MUTAG provides 7 one-hot atom-label features

    - hidden_dim is the number of features learned for each node by a convolution; the established contextual-baseline width is 64

    - num_classes is the number of graph classes, supplied by dataset.num_classes; MUTAG requires 2 output scores per graph

    - These arguments describe feature and class dimensions, not the number of nodes or graphs, so the model can process graphs of different sizes

    - Assigning the layers to self.conv1, self.conv2 and self.classifier registers them as model submodules, making their learned tensors available through model.parameters() to the existing optimiser

- Defined self.conv1 as SAGEConv(num_features, hidden_dim, ...)

    - The first positional argument specifies the input features per node and the second specifies the output features per node

    - On MUTAG, this creates a 7-to-64 transformation for the neighbour aggregate and another 7-to-64 transformation for the node itself

    - Because the initial features are one-hot atom labels, their neighbour mean describes the proportions of the different atom labels among the supplied neighbours

    - Keeping a separate self transformation allows the node's own atom type to contribute differently from the surrounding atom-type distribution

- Defined self.conv2 as SAGEConv(hidden_dim, hidden_dim, ...)

    - This layer receives the 64-feature node representations produced by the first convolution and ReLU, rather than receiving the original 7-feature inputs again

    - It learns its own self and neighbour transformations, each mapping 64 features to 64 features; its parameters are separate from those in conv1

    - Neighbour representations already contain information from the first layer, so the second layer allows information from nodes up to two edges away to influence the receiving node

    - Both layers update every node using the same connectivity; changing feature width does not change the graph's nodes or edges

- Set the scientifically relevant SAGEConv options explicitly in both layers

    - aggr="mean" selects the elementwise mean as the neighbourhood aggregation rule

        - Averaging is independent of neighbour ordering and gives equal weight to each incoming edge entry

        - Dividing by the number of entries controls the aggregate's scale, but loses some multiplicity information: one neighbour with vector [1, 0] and two identical neighbours both produce mean [1, 0]

        - Mean aggregation is the GraphSAGE variant fixed for this contextual baseline

    - root_weight=True includes the separately learned self-node transformation

        - Without it, the layer would use the transformed neighbour aggregate without this explicit contribution from the receiving node

        - It preserves a distinct route for the node's own information rather than averaging it together with its neighbours

    - project=False disables an additional learned transformation and ReLU on neighbour features before aggregation

        - The chosen layer averages the representations supplied to it directly, then transforms their mean

        - This does not disable the learned self and neighbour transformations or the ReLU explicitly applied after each convolution

        - It retains the specified mean-aggregation variant without adding the optional preprocessing operation

    - normalize=False disables L2 normalisation of the convolution's output vectors

        - L2 normalisation would divide each nonzero output vector by its Euclidean length, rescaling it to unit length

        - This option concerns node-vector length, whereas GCN's normalize option concerns degree-based propagation weights

        - The neighbour mean still divides by the number of incoming entries; normalize=False does not turn mean aggregation into sum aggregation

        - Omitting output L2 normalisation is an explicit adaptation to the shared project architecture

    - bias=True adds a learned offset to the combined output

        - PyG places this bias in the neighbour transformation; the self transformation has no separate bias

        - Each convolution therefore learns one bias value per output feature, giving 64 bias parameters

- Retained the original edge_index without inserting additional self loops

    - SAGEConv uses the supplied edges for neighbour aggregation and root_weight=True for its separate self contribution

    - Adding a self loop would also place the receiving node inside its neighbour aggregate, changing the specified operation

    - In the standard source-to-target flow, edge_index[0] identifies sending nodes and edge_index[1] identifies receiving nodes

    - Edge features are not passed to the model, preserving the established categorical-node-feature and connectivity input policy

- Implemented forward using the same structure as the GCN classifier

    - self.conv1(x, edge_index) aggregates and transforms the input node features; F.relu(x) then replaces negative feature values with zero

    - self.conv2(x, edge_index) repeats message passing on those learned representations, followed by another ReLU

    - With project=False, SAGEConv does not supply these post-convolution activations itself; the explicit F.relu calls introduce the nonlinearities between and after the learned graph transformations

    - global_add_pool(x, batch) sums the final node representations separately for each graph, using batch to identify graph membership

    - The batch vector is needed for graph readout; the convolution layers use edge_index, whose disconnected graph components keep messages within their respective graphs

    - self.classifier maps each 64-feature graph representation to 2 raw class logits

    - No final ReLU or softmax is applied because the existing cross-entropy calculation receives raw logits

- Used a graph-classification adaptation of Inductive Representation Learning on Large Graphs

    - The original GraphSAGE motivation is to learn a shared neighbourhood-based function that can generate representations for unseen nodes, rather than learning a separate embedding tied to every node identity

    - The paper's concatenation of self and neighbour vectors followed by a linear transformation can be written as two separate matrix transformations added together, as in PyG

    - This project processes complete graphs and all their supplied neighbours within graph minibatches, without introducing the paper's neighbour-sampling procedure

    - The model omits the output L2 normalisation in the paper's Algorithm 1 and uses the project's global sum readout and graph classifier

    - Two layers, width 64 and ReLU follow the previously fixed contextual-baseline configuration; these are declared project choices rather than a reproduction of the paper's complete experimental system

- Added experiments/models/inspect_graphsage.py using the existing GCN inspection structure and printing style

    - model.named_parameters() exposes the names and tensors of the registered learned parameters

    - parameter.shape shows their dimensions, while parameter.numel() counts the scalar values that contribute to the parameter total

    - In each convolution, lin_l.weight transforms the neighbour mean, lin_l.bias supplies the bias, and lin_r.weight transforms the receiving node

    - The script passes a batch through each operation individually to expose its shape transitions, then calls model(...) separately to exercise the complete forward method

- Ran python -m experiments.models.inspect_graphsage

    - The 32-graph batch contained 585 nodes, producing input shape [585, 7]

    - conv1 changed the shape to [585, 64]; the first ReLU, conv2 and second ReLU each retained [585, 64]

    - Sum pooling produced [32, 64], replacing one row per node with one row per graph

    - The classifier produced [32, 2], and the complete model call returned the same output shape

    - Each first-layer weight matrix had shape [64, 7] and 448 parameters; including its 64-value bias gave 960 parameters

    - Each second-layer weight matrix had shape [64, 64] and 4096 parameters; including its 64-value bias gave 8256 parameters

    - The classifier contained 128 weights and 2 biases, giving 130 parameters and a model total of 9346

    - Weight matrices are stored as [output_features, input_features]; two matrices per convolution explain why matching GCN's feature width does not imply equal parameter counts

    - These observations establish the parameter structure and successful forward-pass dimensions, not the predictive quality of the untrained model

- Updated experiments/train.py to import and construct GraphSAGE and record settings["model"] as "GraphSAGE"

    - The constructor determines which model runs, while the settings entry identifies the model in the saved settings and result filename

    - Reused train_model and evaluate through the existing model(x, edge_index, batch) interface without introducing model-specific training code

    - Retained Adam with learning rate 0.01, weight decay 0.0005, batch size 32, training seed 0 and split seed 0

    - Retained exactly 1000 epochs without early stopping, minimum-validation-loss selection, earliest exact ties and restoration of the cloned selected state

- Ran python -m experiments.train on MUTAG using CUDA

    - Used 150 training graphs, 18 validation graphs and 20 development-test graphs

    - Completed 1000 epochs and selected epoch 33, with validation loss 0.4093 and validation accuracy 0.7778, corresponding to 14 of 18 graphs correct

    - Validation is evaluated every epoch although progress is printed every ten epochs, so epoch 33 could be selected without appearing in the progress lines

    - At epoch 1000, training loss was 0.2735 and validation loss was 0.7341; the improving training loss alongside worse validation loss is consistent with overfitting under the chosen selection criterion

    - Epoch 1000 achieved higher validation accuracy, 0.8889, but its worse cross-entropy meant it was not selected

        - Accuracy depends on which class receives the largest score

        - Cross-entropy also depends on the probability assigned to the correct class, so fewer classification errors can coexist with worse loss when the probability assignments become less favourable overall

    - After restoring epoch 33, development-test loss was 0.4223 and accuracy was 0.8500, corresponding to 17 of 20 graphs correct

    - This is preliminary evidence from one development partition and fit; it does not establish a final model ranking or replace cross-validation

- Saved the development record to results/development_graphsage_mutag_seed0.json

    - The runner reported source commit b20a2eed08b02f1bdf03ad8c9b5c264c53d7a126

    - Total and trainable parameter counts were both 9346, so all registered model parameters remained available for optimisation

    - Training-pass time was 38.9521 seconds, averaging approximately 0.0390 seconds across the 1000 completed epochs

    - The existing timing convention includes loader iteration, device transfers, forward calculation, loss, backward calculation and optimiser updates, with CUDA synchronisation at the boundaries

    - Validation, development-test assessment and result writing are excluded, so this value is not the complete elapsed duration of the command

- 3.2 added GraphSAGE classifier and recorded MUTAG development fit