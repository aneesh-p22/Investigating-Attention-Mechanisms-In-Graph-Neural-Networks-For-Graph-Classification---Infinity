# 4.1 Attention Mathematics

- Studied the single-head attention calculation before implementing GATConv

    - Used Graph Attention Networks, Section 2.1, equations (1)-(4)

    - GAT remains a message-passing model

        - Connectivity determines which messages are permitted

        - Attention calculates coefficients for those messages

    - The calculation transforms node representations, scores eligible receiver-sender pairs, applies LeakyReLU and neighbourhood softmax, and aggregates the weighted transformed vectors

    - The project applies an output bias and ReLU after aggregation

- Defined an attention head

    - One head contains a learned feature transformation and an attention-scoring mechanism

    - Its parameters are shared across the nodes and edges it processes

    - It calculates one weighted neighbourhood representation for each receiving node

    - Attention coefficients are computed during the forward pass rather than stored as independent learned parameters for individual edges

- Defined the notation and dimensions

    - N is the number of nodes

    - F is the input feature count per node

    - F' is the output feature count for one head

    - h_j is the representation of node j, written as a column vector with F coordinates

    - W has shape [F', F]

    - z_j = W h_j is the transformed node representation with F' coordinates

    - i identifies the receiving node and j identifies the sending node

        - alpha_ij weights the message from j to i

    - a_recv and a_send are learned scoring vectors with F' coordinates each

    - b is an output bias with F' coordinates

    - At the first layer, h_j contains the input node features

        - At later layers, it contains the representation produced by the preceding block

- Established a three-node example

    - Nodes 1 and 2 were connected by an undirected edge

    - Nodes 1 and 3 were connected by an undirected edge

    - Nodes 2 and 3 had no direct connection

    - Each undirected edge permitted messages in both directions

    - Included one self connection per node

        - This allows the receiving node's own representation to participate through attention

    - Defined N_tilde(i) as the eligible sending nodes for receiver i, including i itself

        - N_tilde(1) = {1, 2, 3}

        - N_tilde(2) = {1, 2}

        - N_tilde(3) = {1, 3}

    - Nodes outside the receiving neighbourhood do not participate in its attention calculation

        - This restriction is graph masking

- Applied the shared feature transformation

    - Chose F = F' = 2

    - Used h_1 = [1, 0], h_2 = [0, 1] and h_3 = [1, 1]

        - Vectors are written horizontally here for readability, but the calculation treats them as columns

        - These illustrative values have no assigned chemical meaning

    - Chose W = [[1, 1], [0, 1]]

        - The first output coordinate adds the two input coordinates

        - The second output coordinate retains the second input coordinate

    - Applying z_j = W h_j gave:

        - z_1 = [1, 0]

        - z_2 = [1, 1]

        - z_3 = [2, 1]

    - The same W transforms every node within the head

    - The transformed vectors are used for both attention scoring and message aggregation

- Calculated the pair scores for receiver 1

    - Original GAT uses r_ij = a^T [z_i || z_j]

        - || denotes concatenation

        - T denotes transpose

        - Concatenating the receiver and sender vectors produces 2F' coordinates

        - The scoring vector a therefore also has 2F' coordinates

    - Splitting a into receiver and sender parts gives r_ij = a_recv^T z_i + a_send^T z_j

        - This is the same dot product written as two contributions

    - Chose a_recv = [-2, 0] and a_send = [1, 1]

    - For receiver 1, the receiver contribution was -2

    - The sender contributions were one, two and three for nodes 1, 2 and 3 respectively

    - The resulting pair scores were:

        - r_11 = -1

        - r_12 = 0

        - r_13 = 1

    - Each pair score is a scalar and is not yet a normalised attention coefficient

- Applied LeakyReLU inside attention scoring

    - LeakyReLU(t) = t for t >= 0 and 0.2t for t < 0

    - The attention logits are e_ij = LeakyReLU(r_ij)

    - The example logits became:

        - e_11 = -0.2

        - e_12 = 0

        - e_13 = 1

    - The negative branch retains a reduced negative value and has slope 0.2

    - This operation acts on scalar attention scores

        - It is separate from the ReLU applied to node-feature coordinates after the convolution

- Applied softmax over the receiving neighbourhood

    - alpha_ij = exp(e_ij) / sum_{k in N_tilde(i)} exp(e_ik)

    - k runs over eligible senders for the same receiver i

    - Every receiving node has its own denominator

    - For receiver 1, the denominator was exp(-0.2) + exp(0) + exp(1)

        - The exponential values were approximately 0.818731, 1 and 2.718282

        - Their sum was approximately 4.537013

    - The resulting coefficients were:

        - alpha_11 = 0.180456

        - alpha_12 = 0.220409

        - alpha_13 = 0.599135

    - These coefficients sum to one over receiver 1's eligible neighbourhood

    - Receiver 2 instead normalises over {1, 2}

    - Receiver 3 instead normalises over {1, 3}

    - Normalisation is performed separately for each receiver and head

        - It is not performed over graph classes, all edges in the batch or different head columns

    - The normalisation statement applies before coefficient dropout

        - The project uses zero coefficient dropout

- Calculated the weighted neighbourhood aggregate

    - m_i = sum_{j in N_tilde(i)} alpha_ij z_j

    - Each coefficient multiplies the sender's complete transformed vector

    - For receiver 1, the calculation was approximately:

        - m_1 = 0.180456[1, 0] + 0.220409[1, 1] + 0.599135[2, 1]

    - The first coordinate was approximately 1.599135

    - The second coordinate was approximately 0.819544

    - The aggregate was therefore approximately [1.599135, 0.819544]

    - Before bias and activation, this is a convex combination

        - The coefficients are non-negative and sum to one

- Distinguished coefficients from message contributions

    - A larger attention coefficient applies a larger multiplier to its associated transformed vector

    - Comparing different messages also requires considering the values of their transformed vectors

        - For scalar messages, 0.8 × 1 = 0.8

        - A smaller coefficient can produce a larger contribution: 0.2 × 100 = 20

    - A coefficient alone does not establish a neighbour's influence on the final graph prediction

- Applied the output bias and ReLU

    - The project's single-head block computes h'_i = ReLU(m_i + b)

    - Bias is added after aggregation

    - ReLU acts separately on each feature coordinate

        - Negative coordinates become zero

        - Non-negative coordinates are retained

    - Set b = [0, 0] in the example

    - Both aggregate coordinates were positive

    - The updated representation was therefore approximately h'_1 = [1.599135, 0.819544]

    - All node updates in one layer use the representations supplied at the start of that layer

        - Updating another node does not use the newly calculated h'_1

- Connected the mathematics to tensor storage

    - With node vectors stored as rows, H has shape [N, F]

    - Z = H W^T has shape [N, F']

    - Single-head aggregation produces one F'-dimensional output vector per node

    - Bias and ReLU retain shape [N, F']

    - The example uses node labels 1, 2 and 3 for the mathematical explanation

        - Later PyG tensor indices use the usual zero-based numbering

- Connected the calculation to learning

    - W, a_recv, a_send and b are learned parameters

    - Graph-classification gradients can pass through aggregation, softmax and scoring to update them

    - Attention coefficients depend on those parameters, the current representations and the eligible neighbourhood

    - Changing inputs or neighbourhoods can change coefficients even when the learned parameters remain fixed during evaluation

    - Self-attention uses receiving and sending representations from the same collection of node representations

        - It does not mean that a node attends only to itself

- Compared GAT with the contextual aggregation mechanisms

    - GCN uses coefficients determined by connectivity and degree normalisation

        - Its feature transformation is learned

        - Its degree-normalisation coefficients are not calculated from node features

    - For the example graph, degrees after self connections were three, two and two

    - Receiver 1's symmetric GCN coefficients were 1/3, 1/sqrt(6) and 1/sqrt(6)

        - These are approximately 0.333333, 0.408248 and 0.408248

        - Symmetric GCN coefficients do not generally sum to one for each receiving node

    - The example GAT assigned different coefficients to nodes 2 and 3 despite their equal degrees

    - Equal GAT logits would give each of receiver 1's three incoming messages coefficient 1/3

        - Uniform attention produces a neighbourhood mean in this example

        - It does not generally reproduce symmetric GCN normalisation

    - GraphSAGE mean gives equal weight to neighbours when forming the mean

        - Its separate learned self and neighbour transformations remain part of the update

    - GIN sums neighbour representations, adds its self term and applies an MLP

    - These differences concern the aggregation mechanisms rather than changes to the project's shared graph-classifier structure

- Recorded the scope of this substage

    - The numerical features and parameters were deliberately chosen for the worked example

    - The example explains one receiving node's complete single-head update

    - It does not establish trained selectivity, chemical importance or predictive benefit

    - No GAT implementation or development fit was performed in this substage

    - Stage 4.2 will connect the calculation to standard GATConv layers and the common shape-inspection structure

- 4.1 documented the GAT attention calculation





# 4.2 Single-head Implementation and Inspection

- Added the single-head GAT teaching configuration

    - src/models/gat.py defines two GATConv layers, global sum pooling and a linear classifier

    - Both convolutions use one attention head and produce 64 features per node

    - ReLU follows each complete convolution

    - The constructor uses num_features, hidden_dim and num_classes, matching the baseline model interfaces

    - The reference multi-head configuration follows in Stage 4.3

- Connected GATConv to the attention calculation from Stage 4.1

    - Each convolution transforms node features, calculates attention scores, normalises scores over incoming connections and aggregates weighted transformed features

    - lin.weight contains the shared feature transformation matrix W for that layer

    - att_src contains the sender part of the learned attention vector

    - att_dst contains the receiver part of the learned attention vector

    - The sender and receiver contributions are added before applying LeakyReLU and neighbourhood softmax

    - These attention parameters are learned values, whereas the resulting attention coefficients are calculated during each forward pass

    - Each convolution has its own independent learned parameters

- Specified the attention-head options

    - heads=1 selects one attention head in each convolution

    - hidden_dim supplies the output width of that head

    - concat=True concatenates head outputs in the first convolution

    - concat=False averages head outputs in the second convolution

    - With one head, concatenation and averaging both retain that head's 64 output features

- Specified activation and dropout behaviour

    - negative_slope=0.2 controls LeakyReLU inside attention scoring

    - F.relu applies the external activation after each convolution returns its node features

    - dropout=0.0 disables dropout on normalised attention coefficients

    - Feature dropout would act on node-feature values and would require a separate operation

    - No feature dropout is applied

- Specified connectivity and bias behaviour

    - add_self_loops=True includes each receiving node among its eligible senders

    - edge_dim=None selects attention without an edge-feature contribution

    - The forward calls pass node features and connectivity without passing edge features

    - bias=True enables a learned output bias in each convolution

    - The shared feature transformation has no internal bias; the convolution bias is added after aggregation

    - residual=False disables the additional learned skip connection

- Retained the common graph-classification structure

    - forward receives x, edge_index and the node-to-graph assignment tensor batch

    - Both convolution blocks return node representations

    - global_add_pool sums these representations separately for each graph

    - The linear classifier maps each 64-feature graph representation to two class scores on MUTAG

    - The model returns raw logits

- Added experiments/models/inspect_gat.py

    - Loaded MUTAG through the shared load_dataset function

    - Selected the first batch of 32 graphs with shuffle=False

    - Printed the model structure and each named parameter's shape and element count

    - Printed classifier and total parameter counts

    - Inspected shapes after each complete convolution, each ReLU, sum pooling and classification

    - Printed the output shape from a complete model call

    - Used model.eval() and torch.no_grad() for the forward calculations

    - Retained the same variable names, print labels and inspection depth as the baseline inspectors

- Derived the expected parameter shapes

    - The first transformation matrix has shape [64, 7]

    - The second transformation matrix has shape [64, 64]

    - Each sender and receiver attention tensor has shape [1, 1, 64]

    - These attention dimensions represent a leading singleton dimension for broadcasting, one head and 64 channels per head

    - Each convolution output bias has shape [64]

    - The classifier weight has shape [2, 64] and its bias has shape [2]

- Derived the expected parameter counts

    - First convolution: 64 × 7 + 64 + 64 + 64 = 640

    - Second convolution: 64 × 64 + 64 + 64 + 64 = 4,288

    - Classifier: 2 × 64 + 2 = 130

    - Total: 640 + 4,288 + 130 = 5,058

- Established the expected shape transitions

    - For N nodes across the batch, the input shape is [N, 7]

    - Each convolution produces [N, 64]

    - Each ReLU preserves [N, 64]

    - Sum pooling produces [32, 64], changing from one row per node to one row per graph

    - The classifier and complete model call produce [32, 2]

    - These shape observations establish dimensional compatibility; matching shapes alone do not establish numerical equality between the two forward calculations

- Execution

    - Ran python -m experiments.models.inspect_gat

    - Both convolutions used one attention head

    - The classifier contained 130 parameters and the complete model contained 5,058 parameters

    - The input contained 585 nodes with seven features each

    - Both convolutions produced shape [585, 64], which both ReLU operations preserved

    - Sum pooling produced shape [32, 64]

    - The classifier and complete model call both produced shape [32, 2]

    - The observed parameter counts and shapes matched the derived expectations

- 4.2 added GAT and single-head inspection





# 4.3 Reference Multi-head Classifier

- Configured the reference GAT with eight first-layer attention heads

    - Each first-layer head produces eight features per node

    - Concatenating eight heads gives a total first-layer width of 64

    - The second convolution retains one head producing 64 features per node

    - Both convolutions are followed by ReLU

    - Global sum pooling and the linear classifier retain the common graph-classification structure

- Explained the role of multiple heads

    - Each head has its own feature transformation and sender and receiver attention parameters

    - Each head calculates attention coefficients over the eligible senders independently

    - Each head uses its coefficients to aggregate its transformed sender features

    - Heads can learn different transformations and attention patterns, although distinct behaviour is not guaranteed

- Distinguished concatenation from averaging

    - Concatenation joins the output vectors from all heads along the feature dimension

    - Eight vectors containing eight features each become one vector containing 64 features

    - Averaging takes the mean across corresponding output channels from the heads

    - Averaging retains the channel width of one head

    - The first convolution uses concat=True

    - The second convolution uses concat=False with one head, so its output remains that head's 64-feature vector

- Updated the GAT constructor

    - Retained num_features, hidden_dim and num_classes

    - Added heads as an explicit argument controlling the first convolution

    - hidden_dim represents the total first-layer width and the second-layer output width

    - The validation condition requires a positive head count that divides hidden_dim exactly

    - The % operator calculates the remainder used by the divisibility check

    - channels_per_head = hidden_dim // heads calculates the integer channel width of each head

    - The second convolution keeps heads=1 explicitly

- Established the fixed-total-width head rule

    - For H heads and total width D, the per-head width is C = D / H

    - The project fixes D at 64

    - One head uses 64 channels

    - Two heads use 32 channels each

    - Four heads use 16 channels each

    - Eight heads use eight channels each

    - Every configuration produces 64 first-layer output features after concatenation

    - Increasing the number of heads therefore reduces the channels available within each head

    - This defines the later head comparison without running that experiment in this substage

- Connected the head arrangement to the parameter tensors

    - conv1.lin.weight retains shape [64, 7]

    - Its 64 output rows collectively store the eight head transformations

    - Each group of eight output rows supplies one head's transformed features

    - conv1.att_src and conv1.att_dst each change from shape [1, 1, 64] to [1, 8, 8]

    - These dimensions represent a leading singleton dimension for broadcasting, eight heads and eight channels per head

    - Each attention tensor still contains 64 learned values

    - conv1.bias retains shape [64]

    - The second convolution and classifier retain their Stage 4.2 parameter shapes

- Derived the expected parameter count

    - First convolution transformation: 64 × 7 = 448

    - First convolution sender attention: 8 × 8 = 64

    - First convolution receiver attention: 8 × 8 = 64

    - First convolution output bias: 64

    - First convolution total: 640

    - Second convolution total: 4,288

    - Classifier total: 130

    - Complete model total: 5,058

    - The total matches the single-head teaching configuration because the total width is fixed

    - The attention computation changes because the first layer now uses eight separate head calculations

    - Matching parameter counts does not establish equivalent representations or predictive performance

- Updated the existing GAT inspector

    - Passed heads=8 when constructing the model

    - Retained the model, parameter and shape inspection used in Stage 4.2

    - The first convolution is expected to display as GATConv(7, 8, heads=8)

    - The displayed output-channel argument is the width of one head

    - Both convolutions are expected to produce shape [585, 64] for the same first MUTAG batch

    - Sum pooling is expected to produce [32, 64]

    - The classifier and complete model call are expected to produce [32, 2]

- Clarified the project architecture

    - The first-layer reference is eight heads × eight channels = 64 total features

    - The second layer produces node embeddings for graph-level sum pooling

    - The graph classifier produces raw class logits after pooling

    - This is the project's graph-classification adaptation of GAT

    - The shared external activation is ReLU, while attention scoring retains LeakyReLU with negative slope 0.2

- Execution

    - Ran python -m experiments.models.inspect_gat

    - The first convolution displayed GATConv(7, 8, heads=8), confirming eight heads with eight output channels each

    - The second convolution displayed GATConv(64, 64, heads=1)

    - The first-layer sender and receiver attention tensors each had shape [1, 8, 8] and contained 64 parameters

    - The first-layer transformation matrix had shape [64, 7] and contained 448 parameters

    - The classifier contained 130 parameters and the complete model contained 5,058 parameters

    - The input contained 585 nodes with seven features each

    - Both convolutions produced shape [585, 64], which both ReLU operations preserved

    - Sum pooling produced shape [32, 64], giving one representation per graph

    - The classifier and complete model call both produced shape [32, 2], giving two class scores per graph

    - The observed parameter counts and shapes matched the derived expectations

- 4.3 configured the reference multi-head GAT





# 4.4 Attention Coefficient Inspection

- Extended the existing GAT inspector to return attention coefficients

    - Added return_attention_weights=True to both manual convolution calls

    - Each call returns output node features together with an edge index and attention coefficients

    - Retained the common model, parameter, shape-transition and complete-output inspection

    - The model implementation and its trainable parameters are unchanged

- Explained the returned tensors

    - The returned edge index has shape [2, E], where E is the number of connections used after self-loop handling

    - Row 0 identifies sending nodes and row 1 identifies receiving nodes

    - The coefficient tensor has shape [E, H], where H is the number of attention heads

    - Each coefficient row corresponds to the connection in the same column of the returned edge index

    - Each coefficient column corresponds to one head

    - The first convolution has eight coefficient columns and the second has one

- Used the returned connectivity for coefficient inspection

    - GATConv includes self-connections because add_self_loops=True

    - The returned connection count can therefore differ from the original edge count

    - Coefficient rows must be interpreted using the returned edge index

    - The original graph_batch.edge_index remains the connectivity supplied to each convolution

- Selected one receiving neighbourhood

    - Set receiving_node to 0, the first node in the batch

    - Compared the target-node row with receiving_node to create incoming_mask

    - The Boolean mask identifies connections entering the selected node

    - Applied the same mask to edge-index columns and coefficient rows

    - This preserves the correspondence between selected connections and their coefficients

- Displayed the selected connections and coefficients

    - incoming_edges contains the selected source and target indices

    - incoming_edges.t() transposes the edge tensor so each displayed row contains one [source, target] pair

    - incoming_coefficients contains one row per selected incoming connection and one column per head

    - Used the same loop and print structure for both convolutions

    - Labelled the section Attention coefficients (untrained)

- Inspected incoming coefficient sums

    - incoming_coefficients.sum(dim=0) adds across incoming connections separately for each head

    - The result contains eight sums for the first convolution and one for the second

    - Neighbourhood softmax normalises the incoming coefficients separately for each receiver and head

    - With attention dropout disabled, each selected-neighbourhood sum should be approximately one

    - Printed the sums directly without constructing auxiliary tensors or a batch-wide numerical test

    - This inspection concerns the selected receiver and does not verify every node in the batch

- Retained the meaning of each layer's coefficients

    - First-layer coefficients are calculated from the transformed input node features

    - Second-layer coefficients are calculated from the transformed representations entering the second convolution, after the first convolution and ReLU

    - Each layer uses its own learned transformation and attention parameters

    - The inspection uses a randomly initialised model without training

    - Equal coefficients can occur and do not by themselves indicate an implementation error

    - These values illustrate the attention calculation and do not establish learned selectivity or scientific importance

- Established the expected dimensions for the inspected batch

    - The earlier inspection of this batch showed 1,304 original connections and 585 nodes

    - Including one self-loop per node gives 1,889 returned connections

    - Both returned edge indices are expected to have shape [2, 1889]

    - First-layer coefficients are expected to have shape [1889, 8]

    - Second-layer coefficients are expected to have shape [1889, 1]

    - The previously observed incoming senders for node 0 were nodes 1, 5 and itself

    - For these three incoming connections, the selected coefficient shapes are [3, 8] and [3, 1]

    - The incoming sums have shapes [8] and [1]

- Execution

    - Ran python -m experiments.models.inspect_gat

    - The complete model retained 5,058 parameters, including 130 classifier parameters

    - Both convolutions and ReLU operations retained node representation shape [585, 64]

    - Sum pooling produced [32, 64], and the classifier and complete model call produced [32, 2]

    - The original edge index had shape [2, 1304]

    - Both returned edge indices had shape [2, 1889], consistent with adding 585 self-connections

    - First-layer attention coefficients had shape [1889, 8]

    - Second-layer attention coefficients had shape [1889, 1]

    - The selected receiver was node 0

    - Both layers returned incoming connections [1, 0], [5, 0] and [0, 0]

    - The selected first-layer coefficients had shape [3, 8], covering three incoming connections and eight heads

    - The selected second-layer coefficients had shape [3, 1]

    - Every displayed incoming coefficient was approximately 0.3333

    - The first-layer incoming sums displayed eight values of one, and the second-layer sum displayed one value of one

    - These observations matched the expected coefficient dimensions and neighbourhood normalisation for the selected receiver

    - The displayed coefficients were uniform within this selected neighbourhood at the printed precision

    - The inspection did not establish uniform coefficients elsewhere in the batch or explain the equality from the underlying node representations

    - The model was untrained, so these coefficients remain an implementation illustration

- 4.4 added attention coefficient inspection





# 4.5 Development Fit and Trained Attention Example

- Added GAT to the shared model factory

    - Passed num_features, hidden_dim and num_classes through the common constructor interface

    - Passed heads as the additional GAT argument

    - Retained the existing baseline construction branches

- Configured the reference GAT development fit

    - Used MUTAG with the established node-label and connectivity inputs

    - Used two graph convolution layers with total hidden width 64

    - The first convolution used eight heads with eight channels each

    - The second convolution used one head with 64 channels

    - Retained ReLU after both convolutions, global sum pooling and a linear classifier

    - Retained internal LeakyReLU with negative slope 0.2, self-connections, enabled biases and no dropout or residual connection

    - Used Adam with learning rate 0.01 and weight decay 0.0005

    - Used batch size 32, training seed 0 and split seed 0

    - Selected only GAT for this run

- Retained one configuration area

    - settings contains the shared experiment settings

    - model_settings contains the additional arguments required by each model

    - selected_models identifies which models run

    - The runner copies the shared settings and adds the selected model's arguments to construct current_settings

    - GAT's head setting remains available when changing selected_models

    - Baseline run settings do not receive GAT's head argument

- Retained the shared data and training procedure

    - Loaded the dataset through load_dataset with cleaned=False, use_node_attr=False and use_edge_attr=False

    - Used categorical node labels and connectivity without passing edge features to the model

    - Used the existing stratified split with 150 training graphs, 18 validation graphs and 20 development-test graphs

    - Recorded the original dataset indices belonging to each partition

    - Reset the training seed before constructing the model

    - Used a separately seeded generator for training-loader shuffling

    - Trained for 1,000 fixed epochs without early stopping

    - Evaluated validation loss after every training epoch

    - Selected the state with minimum validation cross-entropy, retaining the earliest epoch in an exact tie

    - Restored the selected state before development-test evaluation and model-state saving

- Added selected-state saving for every model

    - Every future run saves a result JSON and a model-state PT file

    - State saving follows the same procedure for every architecture

    - Used save_model_state from src/recording.py

    - model.state_dict() contains the model's parameters and persistent buffers

    - Saving these tensors allows the selected model to be reconstructed later without retraining

    - The saved state does not contain optimiser or random-generator states for resuming training

    - Existing baseline results were preserved without rerunning those experiments

- Explained the paired output paths

    - get_result_path constructs the result JSON filename

    - Replacing .json with .pt produces the corresponding model-state filename

    - This string operation constructs a path and does not rename an existing file

    - prepare_result_path checks both destinations before training begins

    - Exclusive file-writing modes also prevent overwriting when saving occurs

    - model_state_path in the JSON identifies the corresponding saved state

- Fixed the trained attention example before fitting

    - Selected the first graph in the recorded validation partition

    - The saved validation indices identify this graph as dataset index 112

    - Selected receiving node 0 within that graph

    - This selection does not depend on predictions or attention coefficients

    - The separate attention inspector will reconstruct GAT from the recorded settings and load the selected state

    - Attention inspection remains separate from the shared training runner

- Executed the development fit

    - Ran python -m experiments.train

    - Training used CUDA on the NVIDIA GeForce RTX 4070 Laptop GPU

    - Progress output continued through epoch 1,000

    - The final epoch reported training loss 0.2956 and training accuracy 0.8733

    - The final epoch reported validation loss 0.4007 and validation accuracy 0.9444

    - These training metrics were accumulated during the epoch while model parameters changed between batches

- Recorded the selected state

    - Selected epoch 79 after completing the full 1,000-epoch budget

    - Selected validation loss: 0.24233603477478027

    - Selected validation accuracy: 0.9444444444444444

    - This accuracy corresponds to 17 correct predictions among 18 validation graphs

    - Epoch 79 was selected even though progress was printed only every ten epochs

    - Checkpoint selection therefore included epochs between the printed progress lines

    - The selected epoch had lower validation loss than the final epoch, although both reported the same validation accuracy

    - Cross-entropy depends on the probabilities assigned to the correct classes, whereas accuracy counts correct class predictions

    - Equal accuracy therefore does not imply equal validation loss

- Recorded development-test measurements

    - Evaluated the development-test partition using the restored selected state

    - Development-test loss: 0.40577641129493713

    - Development-test accuracy: 0.8

    - This accuracy corresponds to 16 correct predictions among 20 development-test graphs

    - These measurements describe one development split and training seed

    - They do not establish a final model ranking or replace the planned final evaluation

- Recorded parameter counts and training time

    - Total parameters: 5,058

    - Trainable parameters: 5,058

    - The counts matched the reference architecture inspected in Stage 4.3

    - Training time: 47.50632560090162 seconds

    - Mean training time per epoch: 0.047506325600901615 seconds

    - The mean divides the accumulated training time by all 1,000 completed epochs

    - Timing covers the train_epoch calls, including loader iteration, device transfer, forward calculation, loss, backward calculation, optimiser updates and metric calculation and accumulation

    - Validation, development-test evaluation, checkpoint copying and restoration, progress printing, model-state saving and JSON writing are outside this timing

- Confirmed result and model-state writing

    - The terminal reported saving results/development_gat_mutag_seed0.pt

    - The terminal reported saving results/development_gat_mutag_seed0.json

    - Get-Content displayed the saved JSON with settings, partition indices, selection results, development-test measurements, parameter counts, runtime and device information

    - model_state_path identified results/development_gat_mutag_seed0.pt

    - Recorded PyTorch version 2.13.0+cu130, PyTorch Geometric version 2.8.0.post1 and PyTorch CUDA build 13.0

    - Successful state writing was reported, while reloading and using that state remain to be demonstrated by the trained-attention inspector

- Recorded the source-history limitation

    - The JSON recorded source_commit as be7957a6b4926926d323160214948bd1e828f78b

    - This was Git's current HEAD when the run began

    - The Stage 4.5 preparation changes had not been committed before fitting

    - The recorded commit therefore does not by itself identify the complete working-tree source used for this run

    - The combined commit records the preparation changes, result files and learning log after execution

    - Preserved the original source_commit value rather than replacing it with a later commit identifier

    - Future recorded fits retain the requirement to commit their source and settings before execution

- Remaining work

    - Load the saved state in a separate trained-attention inspector

    - Inspect node 0 of validation graph 112 using the same selected-neighbourhood approach as Stage 4.4

    - Record the actual trained coefficients and their interpretation after execution

- 4.5 added shared selected-state saving and recorded the GAT development fit





- Added experiments/models/inspect_gat_attention.py

    - Read the development JSON and obtained the recorded model settings and model_state_path

    - Selected the first graph from the saved validation indices, giving dataset graph 112

    - Retained receiving node 0 as chosen before fitting

    - Used a one-graph DataLoader to retain the graph_batch interface

    - Reconstructed GAT using the recorded hidden width and head count

    - Loaded the saved state dictionary onto the CPU with torch.load

    - Copied the saved parameters and buffers into the model with load_state_dict

    - Used model.eval() and torch.no_grad() for inspection

- Kept the trained inspection focused

    - Retained the shape walkthrough through both convolutions, both ReLU operations, pooling and classification

    - Printed the complete model output shape

    - Inspected the selected receiver's incoming connections, coefficients and per-head sums in both layers

    - Printed tensors directly without edge-by-edge formatting or value conversions

    - Left parameter listings in the general model inspector

    - Omitted whole-graph coefficient listings and repeated selected-node output

- Confirmed saved-state loading and shape transitions

    - Ran python -m experiments.models.inspect_gat_attention

    - The script loaded the recorded state without error and identified selected epoch 79

    - The selected validation graph had 13 nodes with seven input features each

    - Both convolutions produced shape [13, 64], which both ReLU operations preserved

    - Sum pooling produced shape [1, 64]

    - The classifier and complete model call produced shape [1, 2]

    - These outputs confirmed execution of the loaded model on the predetermined graph

- Inspected returned connectivity

    - The original edge index had shape [2, 26]

    - Both convolutions returned edge indices with shape [2, 39]

    - The increase of 13 connections was consistent with adding one self-connection per node

    - Both layers returned incoming connections [1, 0], [5, 0] and [0, 0] for receiver 0

    - Coefficient rows were interpreted using these returned connections

- Recorded first-layer attention

    - The complete coefficient tensor had shape [39, 8]

    - The selected neighbourhood contained three incoming connections and eight heads

    - Every displayed incoming coefficient was approximately 0.3333

    - All eight incoming coefficient sums displayed as one

    - Attention was uniform within this selected neighbourhood at the printed precision

    - The inspection did not establish the reason for this equality or uniformity elsewhere in the graph

- Recorded second-layer attention

    - The complete coefficient tensor had shape [39, 1]

    - The displayed incoming coefficients were 0.0394 for sender 1, 0.9212 for sender 5 and 0.0394 for the self-connection

    - The incoming coefficient sum displayed as 1.0000

    - Sender 5 received the largest coefficient in this neighbourhood

    - The second layer calculated attention from representations entering that layer after the first convolution and ReLU

    - The same connectivity therefore supported different coefficient patterns in the two layers

- Interpreted the trained example

    - Both layers displayed the expected neighbourhood normalisation for the selected receiver

    - A coefficient scales a sender's transformed feature vector within one layer and head

    - The coefficient 0.9212 does not represent a percentage contribution to the final graph prediction

    - This inspection describes one predetermined neighbourhood in one validation graph

    - It does not establish dataset-wide attention behaviour or causal importance

    - The earlier untrained inspection used a different graph, so differences between the examples cannot be attributed solely to training

- Recorded the later analysis planning item

    - Revisit whether dataset-wide attention summaries would improve interpretation before the final experiments

    - If included, define the measurements, graph-selection rules and saved-state requirements in advance

- 4.5 recorded the GAT development fit and inspected attention from its saved selected state