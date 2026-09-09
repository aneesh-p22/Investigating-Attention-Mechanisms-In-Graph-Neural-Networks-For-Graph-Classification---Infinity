# 3.1 Shared Training and Evaluation Functions

- Extracted the existing training and evaluation functions so GraphSAGE and GIN could reuse the GCN training procedure

    - Each model accepts node features, connectivity and graph membership through model(x, edge_index, batch)

    - Each model returns graph-class logits

    - The training code therefore works with the supplied model without needing to know which graph convolution it uses

- Moved train_epoch and train_model into src/training.py

    - train_epoch performs one pass through the training loader and returns graph-mean loss and accuracy

    - train_model calls train_epoch and evaluate after each epoch

    - It preserves the lowest-validation-loss state and restores it after the fixed training budget

    - Its return values are selected epoch, selected validation loss, selected validation accuracy and training-pass seconds

    - These functions receive their inputs through arguments rather than reading the development runner's settings directly

- Moved evaluate into src/evaluation.py

    - Both validation and development-test assessment require the same loss and accuracy calculation without parameter updates

    - src/training.py imports evaluate for validation

    - experiments/train.py imports evaluate for development-test assessment

    - evaluate returns graph-mean loss followed by accuracy

- Updated experiments/train.py to import the extracted functions and removed their original definitions

    - The runner constructs the experiment through its settings, dataset, loaders, model and optimiser

    - It then calls the shared functions for fitting and assessment before recording the result

    - Passing model into train_model gives the function access to the same model object used by the runner

    - Restoring the selected state inside train_model therefore updates the model subsequently assessed by the runner

    - Function bodies, arguments and returned values were preserved during extraction

        - The extraction changed code organisation without changing the intended training procedure

- Selected python -m src.training as the module-loading check

    - This command loads src.training and its src.evaluation import

    - Function definitions are loaded without calling them, so no training or printed results are expected

    - Successful module loading checks the imports

    - It does not establish that a training run produces equivalent numerical results

- Retained the shared training and evaluation conventions

    - graph_batch identifies the complete PyG minibatch

    - logits contains the raw graph-class scores

    - predictions contains the class indices selected by argmax

    - total_loss, total_correct and total_graphs accumulate metrics over individual graphs

    - Training metrics use predictions calculated during successive minibatch updates

    - Validation and development-test metrics evaluate a fixed model state without parameter updates

- Applied the later code-consistency changes to the shared functions

    - Removed the docstrings from train_epoch, train_model and evaluate

    - Retained the explicit function arguments and return values

    - Retained ordinary == for prediction comparisons

    - Retained tensor cloning for selected-state preservation

    - No training-logic change or additional baseline fit was required for these formatting changes

- 3.1 extracted shared training and evaluation functions





# 3.2 GraphSAGE

- Added src/models/graphsage.py to implement a GraphSAGE graph classifier using PyG's SAGEConv operator

    - SAGEConv updates each node by combining information from that node and its neighbours

    - With the chosen settings, it calculates the elementwise mean of the incoming neighbour representations

    - It applies one learned linear transformation to this mean and a separate learned linear transformation to the receiving node's own representation

    - The transformed contributions are added together with a learned bias

    - The operation is W_neighbour × mean(neighbour vectors) + W_self × node vector + bias

        - The model applies ReLU afterwards

    - For example, neighbour vectors [1, 3] and [3, 1] have mean [2, 2]

        - The layer transforms this mean separately from the receiving node's own vector

    - The mean has no learned parameters

        - The two transformations learn how feature coordinates contribute to the output and how self and neighbour information are combined

- Constructed the model with num_features, hidden_dim and num_classes

    - num_features is the number of input features per node

        - It is supplied by dataset.num_node_features

        - MUTAG provides seven one-hot atom-label features

    - hidden_dim is the number of output features per node

        - The established contextual-baseline width is 64

    - num_classes determines the number of graph-class logits

        - It is supplied by dataset.num_classes

        - MUTAG requires two output scores per graph

    - These arguments describe feature and class dimensions rather than the number of nodes or graphs

        - The same model can therefore process graphs of different sizes

    - Assigning the layers to self.conv1, self.conv2 and self.classifier registers them as model submodules

        - Their parameters become available through model.parameters() to the existing optimiser

- Defined self.conv1 as SAGEConv(num_features, hidden_dim, ...)

    - The first positional argument specifies the input features per node

    - The second specifies the output features per node

    - On MUTAG, the layer creates a seven-to-64 transformation for the neighbour mean and another seven-to-64 transformation for the node itself

    - Because the initial features are one-hot atom labels, the neighbour mean describes the proportions of the atom labels among the supplied neighbours

    - The separate self transformation allows the node's own atom type to contribute differently from the surrounding atom-type distribution

- Defined self.conv2 as SAGEConv(hidden_dim, hidden_dim, ...)

    - It receives the 64-feature node representations produced by the first convolution and ReLU

    - It does not receive the original seven-feature inputs again

    - It learns its own self and neighbour transformations, each mapping 64 features to 64 features

        - Its parameters are separate from those in conv1

    - The incoming representations already contain information from the first layer

        - A second message-passing layer allows information from nodes up to two edges away to influence the receiving node

    - Both layers use the supplied connectivity

        - Changing feature width does not change the graph's nodes or edges

- Set the relevant SAGEConv options explicitly in both layers

    - aggr="mean" selects the elementwise mean as the neighbourhood aggregation rule

        - Averaging is independent of neighbour ordering and gives equal weight to each incoming edge entry

        - Dividing by the number of entries controls the aggregate's scale

        - It loses some multiplicity information: one neighbour with vector [1, 0] and two identical neighbours both produce mean [1, 0]

        - Mean aggregation is the GraphSAGE variant fixed for this contextual baseline

    - root_weight=True includes the separately learned self-node transformation

        - Without it, the output would use the transformed neighbour mean without this explicit self contribution

        - The self representation follows a separate learned transformation before being combined with the neighbour contribution

    - project=False disables an additional learned transformation and ReLU on neighbour features before aggregation

        - The chosen layer averages the representations supplied to it directly and then transforms their mean

        - This does not disable the learned self and neighbour transformations

        - It also does not disable the ReLU explicitly applied after the convolution

    - normalize=False disables L2 normalisation of the convolution's output vectors

        - L2 normalisation divides each nonzero vector by its Euclidean length, rescaling it to unit length

        - This option concerns node-vector length

        - GCN's normalize option instead concerns degree-based propagation coefficients

        - The neighbour mean still divides by the number of incoming entries

        - Omitting output L2 normalisation is an explicit adaptation to the shared project architecture

    - bias=True adds a learned offset to the combined output

        - PyG places this bias in the neighbour transformation

        - The self transformation has no separate bias

        - Each convolution therefore has one bias value per output feature, giving 64 bias parameters

- Retained the original edge_index without inserting additional self loops

    - SAGEConv uses the supplied edges for neighbour aggregation

    - root_weight=True provides the separate self contribution

    - Adding a self loop would also include the receiving node in the neighbour mean, changing the specified operation

    - In the standard source-to-target flow, edge_index[0] identifies sending nodes and edge_index[1] identifies receiving nodes

    - Edge features are not passed to the model

        - This preserves the established categorical-node-feature and connectivity input policy

- Implemented forward using the same structure as the GCN classifier

    - self.conv1(x, edge_index) aggregates and transforms the input node features

    - F.relu(x) applies the first external activation

    - self.conv2(x, edge_index) repeats message passing on the learned representations

    - A second F.relu(x) follows the second convolution

    - With project=False, SAGEConv does not supply these external activations itself

        - The explicit ReLU calls introduce nonlinearities between and after the graph transformations

    - global_add_pool(x, batch) sums the final node representations separately for each graph

        - batch identifies graph membership

        - The convolution layers use edge_index, whose disconnected graph components keep messages within their respective graphs

    - self.classifier maps each 64-feature graph representation to two raw class logits

    - No final ReLU or softmax is applied to those logits

        - The shared cross-entropy calculation receives raw class scores

- Used a graph-classification adaptation of Inductive Representation Learning on Large Graphs

    - The original GraphSAGE motivation is to learn a shared neighbourhood-based function that generates representations for unseen nodes

        - It does not require a separately learned embedding tied to every node identity

    - The paper concatenates the self representation and neighbour aggregate before applying a linear transformation

        - Splitting that transformation into self and neighbour blocks gives the equivalent sum of two matrix transformations used by this PyG configuration

    - This project processes complete graphs and all their supplied neighbours within graph minibatches

        - It does not introduce the paper's neighbour-sampling procedure

    - The model omits the output L2 normalisation in the paper's Algorithm 1

    - It uses the project's global sum readout and graph classifier

    - Two layers, width 64 and ReLU are declared project choices

        - They do not constitute a reproduction of the paper's complete experimental system

- Added experiments/models/inspect_graphsage.py using the common model-inspection structure

    - Printed the model to expose the two SAGEConv layers and classifier

    - model.named_parameters() supplied registered parameter names and tensors

    - parameter.shape exposed tensor dimensions

    - parameter.numel() counted scalar entries

    - In each convolution, lin_l.weight transforms the neighbour mean, lin_l.bias supplies the bias and lin_r.weight transforms the receiving node

    - Printed classifier and total parameter counts

    - Passed a batch through each complete convolution, external activation, pooling operation and classifier to expose the shape transitions

    - Called the complete model separately to exercise forward

    - The inspector follows the model's computation without manually reconstructing the neighbour mean or separate transformations

        - Their roles are explained through the operator description and parameter names

- Ran python -m experiments.models.inspect_graphsage

    - The first batch contained 32 graphs and 585 nodes

    - Input shape was [585, 7]

    - conv1 changed the shape to [585, 64]

    - The first ReLU, conv2 and second ReLU each retained [585, 64]

    - Sum pooling produced [32, 64]

        - One row per node was replaced by one row per graph

    - The classifier produced [32, 2]

    - The complete model call returned the same output shape

        - This established successful forward execution with the expected dimensions

        - Matching shapes did not independently establish numerical equality between the manual sequence and complete model call

- Reconciled the parameter counts

    - Each first-layer weight matrix had shape [64, 7] and contained 448 parameters

        - There were two matrices, one for the neighbour mean and one for the self representation

        - Including the 64-value bias gave 448 + 448 + 64 = 960 parameters

    - Each second-layer weight matrix had shape [64, 64] and contained 4,096 parameters

        - Including both matrices and the bias gave 4,096 + 4,096 + 64 = 8,256 parameters

    - The classifier contained 128 weights and two biases

        - Its total was 130 parameters

    - The complete model contained 960 + 8,256 + 130 = 9,346 parameters

    - Weight matrices are stored as output features by input features

    - Two weight matrices per convolution explain why matching GCN's feature width does not imply matching its parameter count

- Updated experiments/train.py to construct and identify GraphSAGE

    - The constructor selected the model implementation

    - settings["model"] was set to GraphSAGE for the saved settings and result filename

    - Reused train_model and evaluate through the existing model(x, edge_index, batch) interface

    - Retained the shared fitting settings

        - Adam learning rate: 0.01

        - Weight decay: 0.0005

        - Batch size: 32

        - Training seed: 0

        - Split seed: 0

        - Epochs: 1000

    - Retained minimum-validation-loss selection, earliest exact ties and restoration of the cloned selected state

    - No early stopping was used

- Ran python -m experiments.train on MUTAG using CUDA

    - Used 150 training graphs, 18 validation graphs and 20 development-test graphs

    - Completed all 1,000 epochs

    - Selected epoch: 33

    - Selected validation loss: 0.4093

    - Selected validation accuracy: 0.7778

        - This corresponds to 14 correct predictions out of 18

    - Epoch 33 was absent from the ten-epoch progress output

        - Its selection demonstrates that selection was not restricted to printed checkpoints

        - The shared training code evaluates validation and updates the selected state after every epoch

- Interpreted the difference between the final and selected states

    - At epoch 1,000, training loss was 0.2735 and validation loss was 0.7341

    - Lower training loss alongside worse validation loss is consistent with overfitting under the chosen selection criterion

    - Epoch 1,000 achieved validation accuracy 0.8889

        - This was higher than the selected state's validation accuracy

        - Its worse cross-entropy meant it was not selected

    - Accuracy depends on which class has the largest score

    - Cross-entropy also depends on the probability assigned to the correct class

        - Fewer classification errors can coexist with worse loss when probability assignments become less favourable overall

- Assessed the restored selected state on the development-test partition

    - Development-test loss: 0.4223

    - Development-test accuracy: 0.8500

        - This corresponds to 17 correct predictions out of 20

    - This is preliminary evidence from one development partition and fit

        - It does not establish a final model ranking or replace cross-validation

- Saved the development record to results/development_graphsage_mutag_seed0.json

    - The runner reported source commit b20a2eed08b02f1bdf03ad8c9b5c264c53d7a126

    - Total and trainable parameter counts were both 9,346

    - Training-pass time was 38.9521 seconds

    - Mean training-pass time was approximately 0.0390 seconds across the 1,000 completed epochs

    - The timer includes the complete train_epoch call

        - Loader iteration, device transfers, forward calculation, loss, backward calculation and optimiser updates

        - Prediction calculation, correct-prediction counting and metric accumulation

        - Final training loss and accuracy calculations

    - CUDA synchronisation ensures that the timed GPU work completes before its elapsed time is recorded

    - Validation, checkpoint copying and restoration, progress printing, development-test assessment and result writing are excluded

        - The recorded time is not the complete duration of the command

- Applied the later inspector-consistency changes

    - Added model.eval() after model construction

    - Wrapped the manual and complete forward calculations in one torch.no_grad() context

    - Standardised the shape labels and separate print() calls for blank output lines

    - Retained torch.relu in the inspector and F.relu in the model

    - These changes preserve the model architecture and parameter counts

    - The numerical observations above remain the original inspection and development-fit evidence

- 3.2 added GraphSAGE classifier and recorded MUTAG development fit





# 3.3 GIN

- Added src/models/gin.py to implement a Graph Isomorphism Network using PyG's GINConv operator

    - GINConv updates each node by summing incoming neighbour representations

    - It adds a weighted contribution from the receiving node itself

    - It passes the combined vector through a neural network

    - The operation is MLP((1 + epsilon) × node vector + sum(neighbour vectors))

        - MLP means multilayer perceptron

    - Neighbourhood summation and self addition occur before the learned transformations inside the MLP

    - GraphSAGE transforms the self and neighbour contributions separately

        - This GIN configuration combines their representations first and transforms the combined vector

- Used sum aggregation to retain information about repeated neighbour features

    - A neighbourhood is a multiset

        - It is an unordered collection in which the same feature vector can occur more than once because different nodes can have identical features

    - Summation is independent of neighbour ordering but includes each occurrence

    - Two neighbours represented by [0, 1] contribute [0, 2]

    - One neighbour represented by [0, 1] contributes [0, 1]

        - Mean aggregation produces [0, 1] in both cases

    - With MUTAG's initial one-hot atom features and epsilon zero, the first aggregation gives atom-type counts across the receiving node and its neighbours

    - Later layers sum learned feature vectors

        - Their coordinates should not be interpreted directly as atom counts

- Constructed the classifier using num_features, hidden_dim and num_classes

    - num_features is supplied by dataset.num_node_features and equals seven for MUTAG

    - hidden_dim is the established contextual-baseline width of 64 features per node

    - num_classes is supplied by dataset.num_classes and equals two

    - GINConv receives a neural network as its first argument rather than separate input and output dimensions

        - The Linear layers inside that network determine its dimensions

- Defined the first convolution's MLP using nn.Sequential

    - nn.Sequential stores the supplied modules and applies them in their written order

        - Each module's output becomes the next module's input

    - nn.Linear(num_features, hidden_dim) maps each aggregated seven-feature vector to 64 features

        - It learns a weight matrix with shape [64, 7]

        - Its default bias=True supplies an additive bias with shape [64]

        - The same transformation is applied to every node's aggregated vector

    - nn.ReLU() applies the elementwise operation max(0, x)

        - It introduces a nonlinearity between the two linear transformations

        - Without this intervening nonlinearity, the two affine transformations could be combined into one affine transformation

        - ReLU has no learned parameters and retains the 64-feature shape

    - nn.Linear(hidden_dim, hidden_dim) learns a second transformation from 64 features to 64 features

        - It has its own weights and bias

    - The first MLP therefore follows seven to 64 to 64 features

        - It learns a nonlinear function of the combined self and neighbourhood information

- Defined the second convolution with a separate nn.Sequential network

    - Its MLP follows 64 to 64 to 64 features

    - It receives the learned node representations produced by the first convolution and external ReLU

    - Its two Linear layers each have a [64, 64] weight matrix and a [64] bias

    - Each convolution owns its own MLP

        - Parameters are not shared between the two graph layers

    - Two GINConv calls perform two rounds of message passing

        - Information from nodes up to two edges away can influence the receiving node

    - The four Linear layers inside the two MLPs do not create four rounds of message passing

        - They transform node vectors without separately exchanging information along graph edges

- Set eps=0.0 and train_eps=False in both GINConv layers

    - eps sets epsilon in the self coefficient 1 + epsilon

    - eps=0.0 includes the receiving node with coefficient one

        - It does not remove the self contribution

    - train_eps=False keeps epsilon fixed during training

        - This gives the GIN-0 variant selected for the project

    - PyG's GINConv uses sum aggregation by default

        - The supplied MLP and epsilon arguments specify the chosen operation

    - Retained the original edge_index without adding self loops

        - GINConv already adds its explicit self term

        - An additional self loop would include the node through the neighbour sum as well, changing its effective contribution

- Distinguished fixed epsilon buffers from learned parameters

    - With train_eps=False, each epsilon tensor is registered as a buffer

    - The epsilon buffer moves with model.to(device) and is included in model.state_dict()

    - It is absent from model.parameters()

        - The Adam optimiser therefore receives the MLP and classifier parameters without receiving epsilon

    - The selected-state cloning and restoration include epsilon because they operate on state_dict()

    - Each convolution has one epsilon value shared across its nodes and feature coordinates

        - It is not a separate coefficient for every node

    - The inspector's buffer section displays this existing model state

        - It does not create or modify the epsilon tensors

- Implemented forward using the common classifier structure

    - self.conv1(x, edge_index) performs neighbour summation, self addition and the first MLP

    - F.relu(x) follows the complete convolution

    - self.conv2(x, edge_index) repeats these operations on the learned representations

    - A second F.relu(x) follows the second convolution

    - The internal nn.ReLU() and external F.relu(x) perform the same activation at different positions

        - nn.ReLU() is a module between the MLP's two Linear layers

        - F.relu(x) acts after the complete convolution, including the MLP's final Linear layer

        - That final linear output can contain negative values

    - global_add_pool(x, batch) sums the final node representations separately for each graph

        - It produces one 64-feature graph vector

    - self.classifier maps each graph vector to two raw class logits

    - No output softmax is added because the shared cross-entropy calculation accepts logits

    - Edge features remain excluded from the model inputs

- Used the GIN formulation from How Powerful Are Graph Neural Networks?, particularly Sections 4.1 and 4.2

    - The paper motivates sum aggregation and MLPs through injectivity

        - An injective function maps distinct inputs to distinct outputs

    - Its expressiveness results depend on assumptions about feature domains, multiset sizes, aggregation and readout functions, and message-passing depth

    - Arbitrary sums of learned vectors are not automatically injective

        - An MLP cannot recover a distinction already lost when its inputs were combined

    - With epsilon fixed at zero, swapping the receiving node's vector with a neighbour's vector leaves their combined sum unchanged

        - The self and neighbour roles are therefore not always distinguishable from that sum

    - This implementation is a two-message-passing-layer GIN-0 adaptation with width 64, external ReLUs and final-layer sum readout

    - It does not concatenate readouts from every layer as proposed in the paper

    - Its construction alone does not establish every theoretical guarantee from the paper or guarantee better classification accuracy

- Added experiments/models/inspect_gin.py using the common inspection structure

    - Printed the model to expose both Sequential networks and their Linear and ReLU modules

    - Used model.named_parameters() to inspect registered parameter names, shapes and element counts

    - Names such as conv1.nn.0.weight identify the first convolution, its internal network and the module at position zero

    - Position one contains ReLU and has no parameters

    - Position two contains the second Linear layer

    - Added model.named_buffers() to display the fixed epsilon tensors separately

    - Printed classifier and total parameter counts

    - Traced the shapes after each complete convolution, external activation, sum pooling and classifier

    - Called the complete model separately to exercise forward

    - The shape walkthrough remains at the same level as the GCN and GraphSAGE inspectors

        - It does not manually reconstruct the neighbour sum or repeat every operation inside the MLP

        - The printed model structure and parameter tensors expose the MLP's internal dimensions

- Ran python -m experiments.models.inspect_gin

    - The first batch contained 32 graphs and 585 nodes

    - Input shape was [585, 7]

    - conv1 produced [585, 64]

    - The first external ReLU, conv2 and second external ReLU retained [585, 64]

    - Sum pooling produced [32, 64]

    - The classifier produced [32, 2]

    - The complete model call also returned [32, 2]

        - This established successful execution with two class logits per graph

        - Matching output shapes did not independently establish numerical equality between the manual sequence and complete forward call

- Reconciled the parameter counts and buffers

    - The first MLP contained:

        - 64 × 7 + 64 = 512 parameters in its first Linear layer

        - 64 × 64 + 64 = 4,160 parameters in its second Linear layer

        - First MLP total: 4,672

    - The second MLP contained:

        - Two Linear layers with 64 × 64 + 64 = 4,160 parameters each

        - Second MLP total: 8,320

    - The classifier contained 2 × 64 + 2 = 130 parameters

    - The complete model contained 4,672 + 8,320 + 130 = 13,122 parameters

    - conv1.eps and conv2.eps each appeared as a buffer with shape [1] and value tensor([0.])

    - These two stored values are excluded from the parameter count because they are buffers

- Updated experiments/train.py to construct and identify GIN

    - The constructor selected the implemented model

    - settings["model"] was set to GIN for the recorded settings and result filename

    - Reused train_model and evaluate through the common model interface

    - Retained MUTAG, width 64, Adam learning rate 0.01, weight decay 0.0005, batch size 32, training seed zero and split seed zero

    - Retained exactly 1,000 epochs without early stopping

    - Retained minimum-validation-loss selection, earliest exact ties and restoration of the cloned selected state

- Ran the GIN development fit through python -m experiments.train using CUDA

    - Used 150 training graphs, 18 validation graphs and 20 development-test graphs

    - Completed all 1,000 epochs

    - Selected epoch: 187

    - Selected validation loss: 0.1833

    - Selected validation accuracy: 0.9444

        - This corresponds to 17 correct predictions out of 18

    - Epoch 187 was absent from the ten-epoch progress output

        - Selection was not restricted to printed checkpoints

        - The shared training code evaluates validation and updates the selected state after every epoch

- Interpreted the final epoch and training fluctuations

    - At epoch 1,000, training loss was 0.1779

    - Training accuracy was 0.9133

    - Validation loss was 0.5461

    - Validation accuracy was 0.8889

    - The final epoch had worse validation loss than the selected epoch

        - Its state was replaced by the preserved epoch-187 state before development-test assessment

    - The printed trajectory included substantial fluctuations

        - Training loss was 1.0930 and validation loss was 1.1731 at epoch 830

        - Lower values followed at epoch 840

    - The printed metrics alone do not identify the precise cause of that fluctuation

    - Later deterioration did not overwrite the selected state

        - Replacement requires strictly lower validation loss

- Assessed the restored selected state on the development-test partition

    - Development-test loss: 0.3646

    - Development-test accuracy: 0.7500

        - This corresponds to 15 correct predictions out of 20

    - Validation and development test concern different small partitions

        - Validation is also used for epoch selection

        - Their score difference does not by itself identify an implementation error

    - These observations remain preliminary development evidence

        - They do not establish final generalisation performance or justify changing the fixed configuration

- Saved the development record to results/development_gin_mutag_seed0.json

    - The runner reported source commit b266b67dcc4d31b483fa188c55dd66d326e09689

    - Total and trainable parameter counts were both 13,122

        - This agreed with the inspection

        - Fixed epsilon buffers were excluded from both counts

    - Training-pass time was 38.5074 seconds

    - Mean training-pass time was approximately 0.0385 seconds per completed epoch

    - The timer includes the complete train_epoch call

        - Loader iteration, device transfers, forward calculation, loss, backward calculation and optimiser updates

        - Prediction calculation, correct-prediction counting and metric accumulation

        - Final training loss and accuracy calculations

    - CUDA synchronisation ensures completion of the timed GPU work

    - Validation, checkpoint copying and restoration, progress printing, development-test assessment and result writing are excluded

        - The value is training-pass time rather than the complete command duration

- Applied the later inspector-consistency changes

    - Added model.eval() after model construction

    - Wrapped the manual and complete forward calculations in one torch.no_grad() context

    - Standardised shape labels and separate print() calls for blank output lines

    - Retained the small epsilon-buffer section as GIN-specific information

    - Kept shape inspection as the primary purpose

        - No additional neighbourhood calculation or numerical-testing machinery was added

    - These changes preserve the model architecture and parameter counts

    - The numerical observations above remain the original inspection and development-fit evidence

- 3.3 added GIN classifier and recorded MUTAG development fit





# 3.4 Shared Model Construction and Development Comparison

- Added src/models/factory.py with build_model(settings, num_features, num_classes)

    - The function reads settings["model"] and uses explicit if/elif branches to construct GCN, GraphSAGE or GIN

    - Each branch passes the input feature count, configured hidden width and number of graph classes into the corresponding constructor

    - The current factory reads settings["hidden_dim"]

        - The original Stage 3.4 version read settings["baseline_width"]

        - The naming change preserves the numerical role of the setting

    - Each call constructs a fresh model

    - An unsupported model name raises ValueError

    - Device placement and optimiser construction remain in the runner

        - The factory's responsibility is model construction

- Organised the runner configuration into three adjacent objects

    - settings contains the shared experiment choices

        - variant: None

        - dataset: MUTAG

        - hidden_dim: 64

        - learning_rate: 0.01

        - weight_decay: 0.0005

        - epochs: 1000

        - batch_size: 32

        - seed: 0

        - split_seed: 0

    - model_settings contains each model's additional arguments

        - The GCN, GraphSAGE and GIN entries are empty dictionaries because these models require no additional configuration fields

        - Attention-specific fields can be added when those models are introduced

        - Shared width, optimiser settings, dataset and split seed remain in settings

    - selected_models contains model names and determines which models run and their execution order

    - These objects form one configuration area at the top of experiments/train.py

        - They do not need to be combined into one flat dictionary

        - Model selection does not require repeatedly deleting and re-entering the stored model-specific settings

- Prepared a separate effective settings dictionary for each selected model

    - settings.copy() creates a dictionary containing the shared values

        - The shared values are scalars or None

        - A shallow copy is sufficient for the updates performed here

        - Updating this copy does not replace values in the original settings dictionary or another run's dictionary

    - current_settings.update(model_settings[model_name]) adds the selected model's additional fields

        - Updating with an empty baseline dictionary adds nothing

        - Model-specific entries are reserved for additional arguments rather than overrides of the shared experimental choices

    - current_settings["model"] = model_name records the model identity

    - The factory, optimiser, training call, filename construction and result record use this effective dictionary

    - Each saved result therefore records the settings for its own fit

        - It does not contain the configurations of unselected models

        - No .pop-based filtering is required

- Added an initial loop to prepare the requested runs before fitting

    - get_result_path(current_settings, result_type) obtains each destination using the existing naming function

    - prepare_result_path(result_path) applies overwrite protection before any selected model begins training

    - Checking destinations first avoids completing an earlier fit before discovering that a later selected model already has a result file

    - run_settings.append(current_settings) retains each prepared dictionary for the training loop

    - get_source_commit() obtains the source identity once for the invocation

        - Each completed result records that identity

- Loaded the dataset and constructed the development partitions once per invocation

    - load_dataset applies the shared dataset-loading policy

    - stratified_split uses the shared split seed

    - Selected models use the same train, validation and development-test index lists

    - Dataset and split seed remain shared choices

        - They are not supplied as model-specific overrides

    - The earlier runner included a check for dataset or split-seed overrides

        - That check was removed in the revised runner because the configuration convention keeps those choices shared

    - Device selection is also shared across the selected fits

- Added the training loop over run_settings

    - set_seed(current_settings["seed"]) resets the ordinary random generators before each model is constructed

    - A fresh torch.Generator is created and seeded for each training loader

        - Its state does not carry over from the preceding fit

    - Fresh training, validation and development-test loaders use the shared partition indices

    - The training loader uses shuffle=True and the dedicated generator

    - The validation and development-test loaders use shuffle=False

    - build_model constructs the selected model

    - .to(device) places its parameters and buffers on the chosen device

    - A fresh Adam optimiser receives the model parameters and shared learning rate and weight decay

    - Each fit begins with its own model, optimiser and training-loader generator while retaining the same data partitions

- Preserved the shared fitting and assessment procedure

    - train_model returns selected epoch, selected validation loss, selected validation accuracy and training-pass seconds

    - Each fit completes exactly 1,000 epochs

    - No early stopping is used

    - Selection uses minimum validation cross-entropy and retains the earliest exact tie

    - The selected state is cloned during fitting and restored afterwards

    - evaluate assesses the restored model on the development-test partition once

    - It returns development-test loss and accuracy

- Preserved the recorded experiment evidence

    - Each result contains its effective settings

    - Dataset and feature policies identify the supplied model inputs

    - Original graph indices identify the three development partitions

    - Selection fields record the criterion, tie rule, completed epochs, selected epoch and selected validation metrics

    - Development-test fields record the assessment of the restored state

    - Parameter fields record total and trainable counts

    - Runtime fields record total training-pass seconds, mean seconds per epoch and the timing convention

    - Device, library-version and source-commit fields preserve execution context

    - save_result writes the result using the existing overwrite protection

- Kept training completion output within experiments/train.py

    - Each fit prints its effective settings and training progress

    - After fitting, it prints the selected epoch and validation metrics

    - It prints development-test metrics, parameter counts and training-pass time

    - It prints the saved result path

    - The earlier accumulated results list and final Completed development fits summary were removed

        - The runner already prints each completed fit

        - Saved-record comparison belongs in experiments/compare.py

    - The normal execution command remains python -m experiments.train

- Kept the runner compatible with the existing shared interfaces

    - Loss is calculated through F.cross_entropy inside the shared functions

        - The runner does not construct and pass a separate criterion object

    - Timing is performed inside train_model around individual training passes

        - The runner does not time the entire train_model call as though it were training-pass time

    - JSON writing is delegated to src/recording.py

        - experiments/train.py constructs Python dictionaries and does not need to import json

    - Reading historical results uses the separate comparison script

        - No argparse flag or run-mode setting is required

- Added experiments/compare.py to read existing development records

    - result_paths explicitly selects the GCN, GraphSAGE and GIN JSON files

    - This list is independent of selected_models in the training runner

        - Historical records can be read without changing which models are selected for future fitting

    - open(result_path, encoding="utf-8") opens each file for reading

    - The with block closes the file afterwards

    - json.load(result_file) converts the JSON contents into Python dictionaries and values

    - The script reads the settings, selection and parameters sections

    - It prints model and dataset identity, training and split seeds, selected epoch, validation metrics, parameter counts and result path

    - Validation accuracy belongs to the selected minimum-validation-loss epoch

        - It is not necessarily the highest validation accuracy reached during training

    - The script performs no model construction, fitting, evaluation or result writing

- Ran python -m experiments.compare

    - All three selected records were read successfully

    - Each reported MUTAG, training seed zero and split seed zero

    - GCN:

        - Selected epoch: 384

        - Validation loss: 0.2568

        - Validation accuracy: 0.8333

        - Total parameters: 4,802

        - Trainable parameters: 4,802

    - GraphSAGE:

        - Selected epoch: 33

        - Validation loss: 0.4093

        - Validation accuracy: 0.7778

        - Total parameters: 9,346

        - Trainable parameters: 9,346

    - GIN:

        - Selected epoch: 187

        - Validation loss: 0.1833

        - Validation accuracy: 0.9444

        - Total parameters: 13,122

        - Trainable parameters: 13,122

    - The displayed records were:

        - results/development_gcn_mutag_seed0.json

        - results/development_graphsage_mutag_seed0.json

        - results/development_gin_mutag_seed0.json

- Recorded what the comparison execution establishes

    - The saved records could be opened, read and displayed successfully

    - The command did not execute the factory or multi-model training loop

    - Matching displayed split seeds did not independently verify equality of the complete stored partition-index lists

    - No additional development fits were run for this comparison

    - The historical JSON files retain their original settings, including baseline_width

        - The reader does not depend on that particular width-field name

        - No rewriting of the original records is required for the current hidden_dim convention

- Interpreted the observations alongside the model differences

    - GIN had the lowest selected validation loss and highest selected validation accuracy among these three recorded fits

    - GCN used the fewest parameters, followed by GraphSAGE and GIN

    - The models share the established inputs, two message-passing layers, width 64, ReLU, sum readout and fitting procedure

    - Their aggregation rules and learned transformations differ

        - GCN uses degree-normalised propagation

        - GraphSAGE combines separate self and mean-neighbour transformations

        - GIN applies MLPs after combining the self representation with neighbour sums

    - Matching feature width does not match parameter count or isolate a single architectural factor

    - The validation observations also come from data used for epoch selection

        - They do not establish final generalisation performance

        - They do not justify choosing a final model winner or changing the shared configuration

- Preserved the boundary between completed fits and later source corrections

    - The baseline inspections and fits retain their recorded outputs and source commits

    - The factory, runner and inspector consistency changes do not create new execution evidence

    - Existing results are retained rather than overwritten or repeated merely to demonstrate formatting changes

    - The next recorded development fit will exercise the revised construction and orchestration code

- 3.4 added shared model construction and development result comparison





# 3 Closing Notes

- Decisions

    - Retain GCN, GraphSAGE and GIN as contextual baselines for the attention investigation

        - All use two message-passing layers

        - Hidden width is 64

        - ReLU follows each convolution block

        - Global sum pooling produces graph representations

        - A linear classifier produces raw graph-class logits

        - Shared dimensions and fitting settings provide consistency without making parameter counts equal or isolating one architectural difference

    - Fix GraphSAGE to the selected mean-aggregation adaptation

        - aggr="mean"

        - root_weight=True

        - project=False

        - normalize=False

        - bias=True

        - Separate learned transformations combine self and neighbour information

        - Full graph minibatches are used without neighbour sampling

        - Output L2 normalisation is omitted as an explicit project adaptation

    - Fix GIN to the selected two-layer GIN-0 adaptation

        - Each convolution contains a two-linear-layer MLP with an internal ReLU

        - An external ReLU follows each complete convolution

        - eps=0.0 and train_eps=False keep the self coefficient at one

        - Epsilon is stored as a buffer

        - Only final-layer node representations are pooled

        - Readouts from every layer are not concatenated

    - Retain the shared fitting procedure

        - Exactly 1,000 epochs

        - No patience or early stopping

        - Minimum-validation-cross-entropy state selection

        - Earliest exact tie

        - Clone and restore the selected state before development-test assessment

        - These decisions supersede older early-stopping instructions in the supplied governing documents

    - Retain MUTAG's established input and development policies

        - Categorical node features and connectivity are used

        - Continuous attributes and edge features are excluded from the principal model inputs

        - The partition contains 150 training, 18 validation and 20 development-test graphs

        - Split seed is zero

        - Training seed is zero

        - Batch size is 32

        - Adam learning rate is 0.01

        - Weight decay is 0.0005

    - Keep each source file's responsibility explicit

        - src/training.py contains the shared fitting functions

        - src/evaluation.py contains the shared assessment function

        - src/models/factory.py constructs the model specified by its effective settings

        - src/recording.py handles result paths, source identity and saving

        - experiments/train.py prepares selected fits, runs them, prints their completion output and records their results

        - experiments/compare.py reads explicitly selected existing records

    - Keep configuration together at the top of the runner using settings, model_settings and selected_models

        - settings contains shared choices

        - model_settings contains additional model-specific arguments

        - selected_models controls which models run and their order

        - Each fit receives a copy of the shared settings with its own additional fields and model identity

        - No .pop-based filtering is used

        - Additional model settings remain stored when a model is temporarily removed from selected_models

    - Give each fit a fresh model, optimiser and seeded training-loader generator

        - Selected models retain the shared dataset and partition indices

        - Result destinations are checked before fitting

        - Existing results are protected from overwriting

    - Preserve the code and inspection style

        - Model constructors use num_features, hidden_dim and num_classes, with additional arguments only where required

        - Model files use import torch.nn as nn and import torch.nn.functional as F

        - Class and function docstrings are omitted

        - Type and return annotations are omitted

        - forward remains a direct sequence without explanatory comments

        - Use ordinary == rather than replacing it with .eq()

        - Keep variable names, multiline-call formatting and blank-line conventions consistent

    - Keep shape inspection as the primary purpose of model inspectors

        - Print the model and registered parameter names, shapes and counts

        - Print classifier and total parameter counts

        - Trace the input, both convolutions, both external activations, sum pooling and classification

        - Call the complete model and print its output shape

        - Use model.eval() and torch.no_grad()

        - Keep the depth and breadth of the common inspection consistent across models

        - Retain small relevant additions such as GCN's pooling toy and GIN's epsilon buffers

        - Connections to the papers are explained through the implemented operations and parameters without reconstructing the convolutions inside the inspectors

    - Retain detailed learning logs where the detail explains consequential behaviour

        - Explain relevant new operations and arguments, their purpose and the meaning of actual output

        - Preserve worked examples and parameter calculations

        - Avoid duplicating full explanations of already established operations

        - Use plain text without inline backtick formatting around identifiers or filenames

        - Separate numbered substage entries with five blank lines

        - Keep the final commit-summary line separate from terminal commit commands

- Ideas

    - Use the mean-versus-sum example in Review 3 to examine what information different aggregators preserve

        - Distinguish retaining multiplicity from guaranteeing an injective representation

    - Revisit how epsilon zero affects the distinction between the receiving node and its neighbours

        - This is a conceptual point for understanding GIN-0

        - It does not activate a change to the fixed model or an additional experiment

    - Extend the existing results reader when final result summarisation becomes necessary

        - Final fold validation and mean and sample-standard-deviation calculations belong to the planned summarisation work

        - Avoid maintaining duplicate historical-results readers

- Report notes

    - The three contextual models have different parameter counts on MUTAG

        - GCN: 4,802

        - GraphSAGE: 9,346

        - GIN: 13,122

        - All counted parameters are trainable

        - GIN's fixed epsilon buffers are excluded from these totals

    - Preserve the development evidence in its existing files

        - results/development_gcn_mutag_seed0.json

        - results/development_graphsage_mutag_seed0.json

        - results/development_gin_mutag_seed0.json

    - The selected-state validation observations were:

        - GCN: epoch 384, loss 0.2568 and accuracy 0.8333

        - GraphSAGE: epoch 33, loss 0.4093 and accuracy 0.7778

        - GIN: epoch 187, loss 0.1833 and accuracy 0.9444

        - These are development observations on validation data used for epoch selection

    - The recorded development-test accuracies were:

        - GCN: 0.8000

        - GraphSAGE: 0.8500

        - GIN: 0.7500

        - Their validation ordering does not establish their development-test ordering or final generalisation ranking

        - The small partitions and single fits limit interpretation

    - GraphSAGE recorded source commit b20a2eed08b02f1bdf03ad8c9b5c264c53d7a126

        - Training-pass time was 38.9521 seconds

        - Mean training-pass time was approximately 0.0390 seconds per epoch

    - GIN recorded source commit b266b67dcc4d31b483fa188c55dd66d326e09689

        - Training-pass time was 38.5074 seconds

        - Mean training-pass time was approximately 0.0385 seconds per epoch

    - Runtime measurements include the complete training pass and its metric calculations

        - They exclude validation, checkpoint copying and restoration, progress printing, development-test assessment and result writing

        - They should not be reported as complete experiment-command durations

    - Distinguish the implemented architectures from the papers' complete systems

        - GraphSAGE uses full graph minibatches and omits the original algorithm's output L2 normalisation

        - GIN uses two message-passing layers, fixed epsilon and final-layer readout

        - The GIN expressiveness results require their stated assumptions

        - The model name and use of summation alone do not establish injectivity or full 1-WL distinguishing power

    - Relevant sources for Review 3 are Inductive Representation Learning on Large Graphs, How Powerful Are Graph Neural Networks? and Neural Message Passing for Quantum Chemistry

        - Connect the implementations to message, aggregation, update and readout operations

        - Explain neighbourhood multisets, injectivity, 1-WL and the limits of architectural comparisons

    - Execution evidence covers the original model inspections, recorded development fits and saved-result comparison

        - The comparison command reads historical records

        - It does not execute the revised factory or multi-model training loop

        - Later naming and formatting changes do not establish new fits or replace historical observations