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





# 3.3 GIN

- Added src/models/gin.py to implement a Graph Isomorphism Network using PyG's GINConv operator

    - GINConv updates each node by summing its incoming neighbour representations, adding a weighted contribution from the receiving node itself, and passing the combined vector through a neural network

    - Its operation is MLP((1 + epsilon) × node vector + sum(neighbour vectors)), where MLP means multilayer perceptron

    - The neighbourhood sum and self addition happen before the learned transformations inside the MLP

    - Unlike GraphSAGE's separate self and neighbour transformations, this configuration combines their representations first and then transforms the combined vector

- Used sum aggregation to retain information about repeated neighbour features

    - A neighbourhood is a multiset: an unordered collection in which the same feature vector can appear more than once because different nodes can have identical features

    - Summation is independent of neighbour ordering but includes every occurrence

    - For example, two neighbours represented by [0, 1] contribute [0, 2], whereas one such neighbour contributes [0, 1]; mean aggregation produces [0, 1] in both cases

    - With MUTAG's initial one-hot atom features and epsilon zero, the first aggregation gives atom-type counts across the receiving node and its neighbours

    - Later layers sum learned feature vectors, so their coordinates should not be interpreted directly as atom counts

- Constructed the classifier using num_features, hidden_dim and num_classes

    - num_features is supplied by dataset.num_node_features and equals 7 for MUTAG

    - hidden_dim is the established contextual-baseline width of 64 features per node

    - num_classes is supplied by dataset.num_classes and equals 2, determining the number of graph-class logits

    - GINConv receives an actual neural network as its first argument rather than separate input and output dimensions; the Linear layers inside that network determine its dimensions

- Defined the first convolution's MLP using nn.Sequential

    - nn.Sequential stores the supplied modules and applies them in their written order, passing each output into the next operation

    - nn.Linear(num_features, hidden_dim) maps each aggregated 7-feature vector to 64 features

        - It learns a weight matrix with shape [64, 7] and, through the default bias=True, an additive bias with shape [64]

        - The same transformation is applied to every node's aggregated vector

    - nn.ReLU() applies the elementwise operation max(0, x), introducing a nonlinearity between the two linear transformations

        - Without this intervening nonlinearity, the two affine transformations could be combined into one affine transformation

        - ReLU has no learned parameters and retains the 64-feature shape

    - nn.Linear(hidden_dim, hidden_dim) learns a second transformation from 64 features to 64 features, with its own weights and bias

    - The complete first MLP therefore follows 7 → 64 → 64 and learns a nonlinear function of the combined self and neighbourhood information

- Defined the second convolution with a separate nn.Sequential network following 64 → 64 → 64

    - It receives the learned node representations produced by the first convolution and external ReLU, rather than receiving the original atom features again

    - Its two Linear layers each have a [64, 64] weight matrix and a [64] bias

    - Each convolution owns its own MLP; parameters are not shared between the two graph layers

    - Two GINConv calls perform two rounds of message passing, allowing information from nodes up to two edges away to influence a node representation

    - The four Linear layers inside the two MLPs do not create four rounds of message passing because they transform node vectors without exchanging information along edges

- Set eps=0.0 and train_eps=False in both GINConv layers

    - eps sets epsilon in the self coefficient 1 + epsilon

    - eps=0.0 therefore includes the receiving node with coefficient one; it does not remove the self contribution

    - train_eps=False keeps epsilon fixed during training, giving the GIN-0 variant specified for the project

    - PyG's GINConv uses sum aggregation by default, so the supplied MLP and epsilon arguments are sufficient to specify this operation

    - Kept the original edge_index without adding self loops because GINConv already adds its explicit self term

        - An additional self loop would also include the node through the neighbour sum, changing its effective contribution

- Distinguished fixed epsilon buffers from learned parameters

    - With train_eps=False, PyG registers each epsilon tensor as a buffer rather than a parameter

    - A buffer is persistent model state that moves with model.to(device) and is included in model.state_dict(), but is absent from model.parameters()

    - The existing Adam optimiser therefore receives the MLP and classifier parameters without receiving epsilon

    - The existing selected-state cloning and restoration also includes these buffers because it operates on state_dict()

    - Each epsilon buffer contains one value, shared across the nodes and feature coordinates processed by that convolution; it is not a separate coefficient for every node

- Implemented forward using the established classifier structure

    - self.conv1(x, edge_index) performs neighbourhood summation, self addition and the first MLP, followed by F.relu(x)

    - self.conv2(x, edge_index) repeats these operations on the learned representations, followed by another F.relu(x)

    - The internal nn.ReLU() and external F.relu(x) perform the same activation at different positions

        - nn.ReLU() is a module placed between the MLP's two Linear layers so that nn.Sequential can execute it

        - F.relu(x) acts after the complete convolution, including the MLP's second Linear layer, whose output can contain negative values

    - global_add_pool(x, batch) sums the final node representations separately for each graph, producing one 64-feature graph vector

    - self.classifier is Linear(hidden_dim, num_classes), mapping each graph vector to two raw class logits

    - No output softmax is added because the shared cross-entropy calculation expects raw logits; edge features remain excluded from the model inputs

- Used the GIN formulation from How Powerful Are Graph Neural Networks?, particularly Sections 4.1 and 4.2

    - The paper motivates sum aggregation and MLPs through injectivity, meaning that distinct inputs produce distinct outputs

    - Its expressiveness results depend on assumptions including countable feature domains, bounded multiset sizes, sufficiently expressive aggregation and readout functions, and sufficient message-passing depth

    - Arbitrary sums of learned vectors are not automatically injective, and an MLP cannot recover a distinction already lost when its inputs were combined

    - With epsilon fixed at zero, swapping the receiving node's vector with a neighbour's vector leaves their combined sum unchanged, illustrating that the self and neighbour roles are not always distinguishable

    - This implementation is a two-message-passing-layer GIN-0 adaptation with width 64, external ReLUs and final-layer sum readout

    - It does not concatenate readouts from every layer as proposed in the paper, and its construction alone does not establish all of the paper's theoretical guarantees or guarantee better classification accuracy

- Added experiments/models/inspect_gin.py using the existing inspection structure and printing style

    - Printed the model to expose both internal Sequential networks and their Linear and ReLU operations

    - Used model.named_parameters() to inspect the learned tensor names, shapes and parameter counts

    - Names such as conv1.nn.0.weight identify the first convolution, its internal network and the module at position 0

    - Position 1 contains ReLU and has no parameters; position 2 contains the second Linear layer

    - Added model.named_buffers() to inspect the fixed epsilon tensors separately from the parameters

    - Traced the convolution, activation, pooling and classifier shapes individually, then called model(...) separately to exercise the complete forward method

- Ran python -m experiments.models.inspect_gin

    - The first batch contained 32 graphs and 585 nodes, giving input shape [585, 7]

    - conv1 produced [585, 64]; the first external ReLU, conv2 and second external ReLU retained [585, 64]

    - Sum pooling produced [32, 64], and the classifier produced [32, 2]

    - The complete model call also returned [32, 2], confirming a successful forward pass with two logits per graph

    - The first MLP contained (64 × 7 + 64) + (64 × 64 + 64) = 4672 parameters

    - The second MLP contained 2 × (64 × 64 + 64) = 8320 parameters

    - The classifier contained 2 × 64 + 2 = 130 parameters, giving 13122 parameters in total

    - conv1.eps and conv2.eps each appeared as a buffer with shape [1] and value tensor([0.])

    - These two stored values are excluded from the parameter count because they are buffers, not learned parameters

- Updated experiments/train.py to import and construct GIN and record settings["model"] as "GIN"

    - The constructor selects the implemented model, while the settings entry identifies it in the recorded settings and result filename

    - Reused train_model and evaluate through the same model(x, edge_index, batch) interface

    - Retained MUTAG, width 64, Adam learning rate 0.01, weight decay 0.0005, batch size 32, training seed 0 and split seed 0

    - Retained exactly 1000 epochs without early stopping, minimum-validation-loss selection, earliest exact ties and restoration of the cloned selected state

- Ran the GIN development fit through python -m experiments.train using CUDA

    - Used 150 training graphs, 18 validation graphs and 20 development-test graphs

    - Completed all 1000 epochs and selected epoch 187

    - Selected validation loss was 0.1833 and validation accuracy was 0.9444, corresponding to 17 of 18 validation graphs correct

    - Epoch 187 could be selected even though it was absent from the printed progress lines because validation is evaluated every epoch and progress is printed every ten epochs

    - At epoch 1000, training loss was 0.1779, training accuracy was 0.9133, validation loss was 0.5461 and validation accuracy was 0.8889

    - The final epoch had worse validation loss than the selected epoch, so its parameters were replaced by the preserved epoch-187 state before development-test assessment

    - The printed trajectory showed substantial fluctuations, including training loss 1.0930 and validation loss 1.1731 at epoch 830, followed by lower values at epoch 840

        - Training was not monotonic; the printed metrics alone do not identify the precise cause of that fluctuation

        - Later deterioration did not overwrite the selected state because replacement requires strictly lower validation loss

    - After selected-state restoration, development-test loss was 0.3646 and accuracy was 0.7500, corresponding to 15 of 20 graphs correct

    - The validation and development-test scores concern different small partitions, with validation also used for epoch selection; their difference does not by itself identify an implementation error

    - This remains preliminary evidence from one development partition and fit, not final cross-validation evidence or a basis for changing the fixed configuration

- Saved the development record to results/development_gin_mutag_seed0.json

    - The runner reported source commit b266b67dcc4d31b483fa188c55dd66d326e09689

    - Total and trainable parameter counts were both 13122, consistent with the inspection and the fixed epsilon buffers being excluded

    - Training-pass time was 38.5074 seconds, averaging approximately 0.0385 seconds per completed epoch

    - The established timing convention includes loader iteration, device transfers, forward calculation, loss, backward calculation and optimiser updates, with CUDA synchronisation at the boundaries

    - Validation, development-test assessment and result writing are excluded, so this is training-pass time rather than the complete command duration

- 3.3 added GIN classifier and recorded MUTAG development fit





# 3.4 Shared Model Construction and Development Comparison

- Added src/models/factory.py with build_model(settings, num_features, num_classes)

    - The function reads settings["model"] and uses explicit if/elif branches to construct GCN, GraphSAGE or GIN

    - Each branch passes the dataset's input feature count, settings["baseline_width"] and the number of graph classes into the corresponding constructor

    - The supplied settings therefore determine both the model's identity and its configured width

    - Every call constructs a fresh model rather than returning an existing model with previously learned parameters

    - An unsupported model name raises ValueError instead of silently constructing a different model

    - Device placement and optimiser construction remain in the training runner because the factory's responsibility is model construction

- Organised experiments/train.py around shared settings, model-specific settings and a list of selected model names

    - settings contains the shared defaults, including dataset, width, optimiser settings, epoch budget, batch size and seeds

    - model_settings contains a dictionary for each model's additional settings or overrides

        - The GCN, GraphSAGE and GIN entries are currently empty because they use the shared defaults

        - Later model-specific fields can be added to the relevant entries without placing irrelevant attention settings in non-attention model records

    - selected_models contains only model names and determines which models are trained and their execution order

    - This replaces repeatedly changing the runner's model import and constructor when moving between models

- Prepared a separate effective settings dictionary for each selected model

    - settings.copy() creates a new dictionary containing the shared values

        - The current values are scalars or None, so a shallow dictionary copy is sufficient for the updates performed here

        - Changing one model's dictionary does not replace values in the shared settings dictionary or another model's dictionary

    - current_settings.update(model_settings[model_name]) adds model-specific fields and replaces shared values where the same key is explicitly supplied

    - current_settings["model"] = model_name records the selected identity after applying the overrides

    - The factory, optimiser, training call, filename construction and result record all use this effective dictionary

    - Saving current_settings records the settings used for that individual fit rather than the complete collection of configurations for other models

- Added an initial loop to prepare all requested runs before fitting

    - get_result_path(current_settings, result_type) obtains each run's destination using the existing naming function

    - prepare_result_path(result_path) applies the existing overwrite protection before any selected model starts training

    - Checking all destinations first avoids completing an earlier fit before discovering that a later selected run already has a result file

    - run_settings.append(current_settings) retains each prepared dictionary for the subsequent training loop

    - get_source_commit() obtains the source identity once for the invocation, and each result records that identity

- Loaded the dataset and constructed the development partitions once per training invocation

    - All selected models use the same dataset and the same train, validation and development-test index lists

    - The dataset and split seed remain shared settings rather than model-specific experimental choices

    - The runner checks that the effective dataset and split seed still match the shared values

        - If either differs, it raises ValueError because the already prepared dataset and partitions would not match the settings being recorded for that model

    - Device selection is also shared across the selected fits

- Added the training loop over the prepared run_settings dictionaries

    - set_seed(current_settings["seed"]) resets the ordinary random generators before constructing each model

    - A new torch.Generator is created and seeded for each training loader, preventing the shuffle-generator state from carrying over from the preceding fit

    - Fresh training, validation and test loaders are constructed using the shared partition indices and the effective batch size

    - build_model(current_settings, ...) creates the selected model, and .to(device) places its parameters and buffers on the chosen device

    - A fresh Adam optimiser receives that model's parameters and the effective learning rate and weight decay

    - Each fit therefore begins with its own model, optimiser and loader generator while retaining the shared data partitions

- Preserved the established training, selection and recording behaviour

    - The shared train_model still returns selected epoch, selected validation loss, selected validation accuracy and training-pass seconds

    - The configured budget remains exactly 1000 epochs without early stopping

    - Selection still uses minimum validation cross-entropy, retaining the earliest exact tie and restoring the cloned selected state

    - evaluate assesses the restored model on the development-test partition once and returns loss and accuracy

    - Each result retains its effective settings, input policy, partition indices, selection details, development-test metrics, parameter counts, runtime convention, device and source commit

    - save_result(result_path, result) writes each completed fit through the existing recording function

- Added a final summary of the fits completed during the current training invocation

    - results.append(result) retains each completed fit's existing result dictionary after it has been saved

    - The final loop reads those dictionaries to print the model, selected epoch, validation loss and accuracy, development-test accuracy, parameter count and mean training seconds per epoch

    - It does not repeat model evaluation or calculate new predictions

    - The in-memory results list contains only the fits from that invocation; the saved JSON files preserve results across separate executions

- Kept experiments/train.py focused on training and its resulting outputs

    - Its main guard calls main(), and the normal training command remains python -m experiments.train

    - Reading historical records is handled separately, so no argparse flag or run-mode setting is needed

    - train.py does not need import json because it constructs Python dictionaries and delegates JSON writing to save_result in src/recording.py

- Added experiments/compare.py to read existing development records

    - result_paths explicitly selects the saved GCN, GraphSAGE and GIN JSON files

    - Keeping this list separate from the training settings makes the chosen historical records independent of later changes to selected_models or experiment settings

    - open(result_path, encoding="utf-8") opens each file for reading, and the with block closes it afterwards

    - json.load(result_file) converts the JSON contents into Python dictionaries and values

    - The script reads the saved settings, selection and parameters sections and prints their relevant fields

    - Validation accuracy is the accuracy at the selected minimum-validation-loss epoch, not necessarily the highest validation accuracy reached during training

    - This script performs no model construction, training, evaluation or result writing, so it does not duplicate the training runner

- Ran python -m experiments.compare

    - All three selected records were read successfully and reported MUTAG, training seed 0 and split seed 0

    - GCN selected epoch 384, with validation loss 0.2568, validation accuracy 0.8333 and 4802 parameters

    - GraphSAGE selected epoch 33, with validation loss 0.4093, validation accuracy 0.7778 and 9346 parameters

    - GIN selected epoch 187, with validation loss 0.1833, validation accuracy 0.9444 and 13122 parameters

    - Total and trainable parameter counts matched for each model

    - The displayed records were results/development_gcn_mutag_seed0.json, results/development_graphsage_mutag_seed0.json and results/development_gin_mutag_seed0.json

    - This execution establishes successful reading and display of the saved records; it does not execute the new factory or multi-model training loop, or independently verify equality of the stored partition indices

    - No additional development fits were run for this comparison

- Interpreted the development observations alongside the model differences

    - GIN had the lowest selected validation loss and highest selected validation accuracy among these three recorded fits

    - GCN used the fewest parameters, followed by GraphSAGE and GIN

    - The models share the established inputs, two message-passing layers, width 64, ReLU, sum readout and training procedure, but their learned transformations and aggregation rules differ

    - GCN uses degree-normalised propagation, GraphSAGE combines separate self and mean-neighbour transformations, and GIN applies MLPs after combining self and neighbour sums

    - These differences mean that matching feature width does not match parameter count or isolate the effect of one architectural factor

    - The observations come from development validation data used for epoch selection; they do not establish final generalisation performance or justify choosing a winner or changing the fixed settings

- 3.4 added shared model construction and development result comparison





# 3 Closing Notes

- Decisions

    - Retained GCN, GraphSAGE and GIN as contextual baselines for the later attention investigation

        - All use two message-passing layers, width 64, ReLU after each block, global sum readout and a linear graph classifier

        - Shared widths and training settings provide consistency without making the models identical in parameter count or isolating a single architectural difference

    - Fixed GraphSAGE to mean aggregation with root_weight=True, project=False, normalize=False and bias=True

        - Separate learned transformations combine self and neighbour information

        - Full graph minibatches replace neighbour sampling, and output L2 normalisation is omitted as an explicit project adaptation

    - Fixed GIN to the two-layer GIN-0 adaptation

        - Each convolution contains a two-linear-layer MLP with an internal ReLU, followed by the external ReLU

        - eps=0.0 and train_eps=False keep the self coefficient at one; epsilon is stored as a buffer

        - Only the final node representations are pooled, without concatenating readouts from every layer

    - Retained the established training procedure

        - Exactly 1000 epochs, no early stopping, minimum-validation-cross-entropy selection and the earliest exact tie

        - Clone the selected state and restore it before development-test assessment

        - This explicit decision takes precedence over the older early-stopping instructions in the supplied governing documents

    - Retained MUTAG's established input and development policies

        - Categorical node features and connectivity are used; edge features and continuous attributes are excluded

        - The development partition contains 150 training, 18 validation and 20 test graphs, using split seed 0

        - Training seed 0, batch size 32, Adam learning rate 0.01 and weight decay 0.0005 remain fixed for these contextual models

    - Assigned each file a clear responsibility

        - src/training.py and src/evaluation.py contain the shared fitting and assessment functions

        - src/models/factory.py constructs the model specified by its effective settings

        - experiments/train.py prepares and trains selected models, saves each result and summarises the fits completed in that invocation

        - experiments/compare.py reads explicitly selected existing records without duplicating training or evaluation

        - The separate results reader is an intentional refinement of the roadmap's earlier suggestion to place the comparison in the training runner

    - Adopted shared settings, model_settings and a selected_models list containing only model names

        - Each fit receives a separate effective settings dictionary used consistently for construction, optimisation and recording

        - Model-specific settings will be added when their models are introduced, keeping irrelevant attention fields out of contextual-model records

        - Each selected fit receives a fresh model, optimiser and seeded training-loader generator

        - Existing result destinations are checked before fitting; completed development fits are not repeated merely to print a comparison

    - Preserved the agreed implementation and teaching style

        - Follow the existing model and inspection scripts, without type annotations, return annotations, explanatory comments inside forward or unnecessary conversions

        - Keep terminal commands short and use ordinary python -m module commands

        - Logs explain all relevant nontrivial new operations and arguments, their purpose, technical behaviour and actual output interpretation, while avoiding trivial syntax and repeated teaching

- Ideas

    - Use the mean-versus-sum example in Review 3 to examine what information different neighbourhood aggregators preserve

        - Distinguish retaining multiplicity from guaranteeing an injective representation

    - Revisit how epsilon zero affects the distinction between a receiving node and its neighbours

        - This is a conceptual point for understanding GIN-0, not an instruction to change the fixed model or add another experiment

    - When final result summarisation becomes necessary, extend or consolidate the existing results reader rather than maintaining duplicate reading and formatting code

        - Final fold validation and mean/sample-standard-deviation calculations belong to the planned summarisation work

- Report notes

    - The three implemented contextual models have different parameter counts on MUTAG

        - GCN: 4802

        - GraphSAGE: 9346

        - GIN: 13122

        - All counted parameters are trainable; GIN's fixed epsilon buffers are excluded from these totals

    - Preserve the development evidence in its existing records

        - results/development_gcn_mutag_seed0.json

        - results/development_graphsage_mutag_seed0.json

        - results/development_gin_mutag_seed0.json

    - The saved selected-state validation observations were

        - GCN: epoch 384, loss 0.2568 and accuracy 0.8333

        - GraphSAGE: epoch 33, loss 0.4093 and accuracy 0.7778

        - GIN: epoch 187, loss 0.1833 and accuracy 0.9444

        - These are development observations on validation data used for epoch selection, not final cross-validation findings

    - The recorded GraphSAGE and GIN development-test accuracies were 0.8500 and 0.7500 respectively

        - Their validation ordering does not establish their development-test ordering or final generalisation ranking

        - The small partitions and single fits limit interpretation; no configuration was selected or changed from these observations

    - Distinguish the implemented architectures from their source papers' complete systems

        - GraphSAGE uses full graph minibatches and omits the original algorithm's output L2 normalisation

        - GIN uses two message-passing layers, fixed epsilon and final-layer readout

        - The GIN expressiveness results require their stated assumptions; the model name and use of summation alone do not establish injectivity or full 1-WL distinguishing power

    - Relevant sources for Review 3 are Inductive Representation Learning on Large Graphs, How Powerful Are Graph Neural Networks? and Neural Message Passing for Quantum Chemistry

        - Connect the implemented models to message, aggregation, update and readout operations

        - Explain neighbourhood multisets, injectivity, 1-WL and the limits of architectural comparisons

    - Execution evidence covers the model inspections, recorded development fits and saved-result comparison

        - The comparison command reads historical records and does not exercise the new Stage 3.4 factory or multi-model training loop

        - Those new orchestration changes have not yet been exercised by a supplied training output; the next planned development fit will use them