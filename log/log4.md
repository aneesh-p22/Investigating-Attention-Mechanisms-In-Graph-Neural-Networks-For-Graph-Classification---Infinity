# 4.1 Attention Mathematics

- Studied the Graph Attention Network mechanism before implementing GATConv

    - Used Graph Attention Networks, Section 2.1, equations (1)-(4), for the single-head calculation

    - GAT remains a message-passing model: graph connectivity determines which messages are permitted, and attention calculates their relative weights

    - The calculation transforms node representations, scores eligible source-receiver pairs, applies LeakyReLU and neighbourhood softmax, then aggregates the weighted transformed vectors

- Connected attention to the previously implemented aggregation mechanisms

    - GCN uses neighbour coefficients derived from graph connectivity and degree normalisation

        - Its feature transformation is learned, but its degree-normalisation coefficients are not calculated from node features

    - GraphSAGE mean assigns equal coefficients to neighbours when forming the neighbourhood mean

        - Its separate learned self and neighbour transformations remain part of the complete update

    - GIN sums neighbouring representations with equal coefficients, combines this sum with the self term and applies its MLP

    - GAT calculates coefficients from node representations and learned scoring parameters

        - Equal coefficients do not imply equal numerical message contributions, because the vectors being weighted can differ

- Defined an attention head

    - One head is one separately parameterised feature transformation and attention-scoring mechanism

    - It calculates one weighted neighbourhood representation for every receiving node

    - Its parameters are shared across nodes and edges within the head

    - It does not contain a separate learned transformation or independently stored attention coefficient for every node or edge

- Defined the notation and dimensions

    - N is the number of nodes, F is the input feature count and F' is the output feature count for one head

    - h_j is the current representation of node j, written mathematically as a column vector with F coordinates

    - W has shape [F', F], and z_j = W h_j has F' coordinates

    - i identifies the receiving node and j identifies the sending node

        - alpha_ij weights the message from j to i

        - Reversing the direction describes a different receiving-node calculation

    - a_recv and a_send are learned scoring vectors, each with F' coordinates

    - b is an output bias with F' coordinates

    - At the first layer, h_j contains the input node features; at later layers, it contains the previous block's representation

- Established the graph and self-loop convention for a numerical example

    - Used three nodes, with undirected connections between nodes 1 and 2 and between nodes 1 and 3

        - Nodes 2 and 3 have no direct connection

        - Each undirected connection permits messages in both directions

    - Included one self loop per node

        - The self loop allows the receiving node's own representation to participate through the same attention mechanism as its neighbours

    - Defined N_tilde(i) as the eligible sending nodes for receiver i, including i itself

        - N_tilde(1) = {1, 2, 3}

        - N_tilde(2) = {1, 2}

        - N_tilde(3) = {1, 3}

    - This restriction is graph masking: nodes outside the receiving neighbourhood do not participate in its local attention calculation

    - Self-attention means the receiving and sending representations come from the same collection of node representations; it does not mean that a node attends only to itself

- Applied the shared learned feature transformation

    - Chose F = F' = 2 and used h_1 = [1, 0], h_2 = [0, 1] and h_3 = [1, 1]

        - Vectors are written horizontally here for readability, but the calculation treats them as columns

        - These are illustrative features with no assigned chemical meaning

    - Chose W = [[1, 1], [0, 1]]

        - Its first output coordinate adds the two input coordinates

        - Its second output coordinate retains the second input coordinate

    - Applying z_j = W h_j gave z_1 = [1, 0], z_2 = [1, 1] and z_3 = [2, 1]

    - The same W is applied to every node in the head

        - Its entries would be updated through backpropagation during training

        - The transformed vectors are used both for attention scoring and as the messages being aggregated

- Calculated the unnormalised pair scores

    - Original GAT forms r_ij = a^T [z_i || z_j], where || denotes concatenation and T denotes transpose

        - Concatenation joins the receiver and sender vectors into one vector with 2F' coordinates

        - The learned scoring vector a also contains 2F' coordinates

        - Their dot product produces one scalar before LeakyReLU

    - Splitting a into receiver and sender parts gives the equivalent expression r_ij = a_recv^T z_i + a_send^T z_j

        - W determines the transformed node representation

        - The scoring vectors determine how that representation contributes to the pair score

    - Chose a_recv = [-2, 0] and a_send = [1, 1]

        - For receiver 1, a_recv^T z_1 = -2

        - The sender contributions are 1, 2 and 3 for nodes 1, 2 and 3 respectively

        - The resulting pair scores are r_11 = -1, r_12 = 0 and r_13 = 1

    - These scalar scores are not yet normalised attention weights

- Applied LeakyReLU inside the attention calculation

    - Defined L(t) = t for t >= 0 and L(t) = 0.2t for t < 0

    - The attention logits are e_ij = L(r_ij)

        - The example logits became e_11 = -0.2, e_12 = 0 and e_13 = 1

    - The negative branch retains a reduced negative value and has derivative 0.2

        - It therefore permits a gradient through that branch of the score calculation

    - This LeakyReLU acts on scalar scores and is separate from the ELU applied after the convolution

- Applied softmax over each receiving neighbourhood

    - alpha_ij = exp(e_ij) / sum_{k in N_tilde(i)} exp(e_ik)

        - k runs over eligible senders for the same receiver i

        - Every receiver has its own denominator

        - Normalisation is not performed globally over every edge, node or graph in a batch

    - For receiver 1, the denominator was exp(-0.2) + exp(0) + exp(1)

        - The exponential values were approximately 0.818731, 1 and 2.718282

        - Their sum was approximately 4.537013

    - Dividing by this denominator gave alpha_11 = 0.180456, alpha_12 = 0.220409 and alpha_13 = 0.599135, approximately

        - These coefficients are positive and sum to one over receiver 1's eligible neighbourhood

        - Receiver 2 instead normalises over {1, 2}, while receiver 3 normalises over {1, 3}

    - The coefficients distribute weight across incoming messages rather than graph classes

    - The normalisation statement applies before coefficient dropout; the project uses zero coefficient dropout

- Calculated the weighted neighbourhood aggregate

    - Defined m_i = sum_{j in N_tilde(i)} alpha_ij z_j

        - Each coefficient is a scalar multiplying the sender's entire F'-dimensional transformed vector

    - For receiver 1, m_1 = 0.180456[1, 0] + 0.220409[1, 1] + 0.599135[2, 1], using rounded coefficients

        - The first coordinate is approximately 0.180456 + 0.220409 + 2(0.599135) = 1.599135

        - The second coordinate is approximately 0.220409 + 0.599135 = 0.819544

        - The aggregate is therefore approximately [1.599135, 0.819544]

    - Before bias and activation, this is a convex combination

        - A convex combination is a weighted sum whose weights are non-negative and sum to one

- Distinguished an attention coefficient from the resulting message contribution

    - A larger coefficient applies a larger multiplier to the associated transformed representation

    - Comparing different neighbours also requires considering the magnitudes and directions of their transformed vectors

        - For scalar messages, 0.8 × 1 = 0.8, while 0.2 × 100 = 20

        - The smaller coefficient can therefore accompany the larger message contribution

    - A coefficient alone does not establish a neighbour's influence on the final graph prediction

- Applied the output bias and node activation

    - The project's single-head block computes h'_i = ELU(m_i + b)

        - Equivalently, h'_i = ELU(sum_{j in N_tilde(i)} alpha_ij W h_j + b)

        - The bias is added after aggregation, and ELU acts separately on each coordinate

    - With its scale parameter equal to one, ELU(t) = t for t > 0 and ELU(t) = exp(t) - 1 for t <= 0

    - Set b = [0, 0] in the numerical example

        - Both aggregate coordinates were positive, so ELU left them unchanged

        - The updated representation was h'_1 approximately equal to [1.599135, 0.819544]

    - Every node update in one layer uses the same input-layer representations

        - Updating another node during that layer does not use the newly calculated h'_1

- Connected the mathematics to tensor storage and learning

    - With node vectors stored as rows, H has shape [N, F]

        - Z = H W^T has shape [N, F']

        - Single-head aggregation and the elementwise activation retain one F'-dimensional output row per node

    - W, a_recv, a_send and b are learned parameters

        - Graph-classification gradients can pass through aggregation, softmax and scoring to update them

    - Attention coefficients are calculated from the parameters, current representations and eligible neighbourhood

        - They are not independent stored parameters for individual edges

        - Even with fixed parameters during evaluation, changing inputs or neighbourhoods can change the coefficients

- Compared the example with symmetric GCN normalisation

    - On an unweighted undirected graph, GCN assigns an eligible message the coefficient c_ij = 1 / sqrt(d_tilde_i d_tilde_j)

        - d_tilde_i is the degree after self-loop inclusion

    - The example degrees were 3, 2 and 2

        - Receiver 1's GCN coefficients were 1/3, 1/sqrt(6) and 1/sqrt(6)

        - These are approximately 0.333333, 0.408248 and 0.408248

    - The example GAT assigned different coefficients to nodes 2 and 3 despite their equal degrees

    - Symmetric GCN coefficients do not generally sum to one for each receiver

    - Equal GAT logits give alpha_ij = 1 / |N_tilde(i)| in this simple graph

        - |N_tilde(i)| is the number of eligible senders

        - Receiver 1 would receive coefficients 1/3, 1/3 and 1/3, producing a neighbourhood mean

        - Uniform GAT attention therefore does not generally reproduce symmetric GCN normalisation

- Recorded the evidence boundary

    - All numerical features and parameters were deliberately chosen for the worked example

    - The calculation illustrates the mechanism rather than trained selectivity, chemical importance or predictive benefit

    - No project source implementation or development fit was performed in this substage

- Commit: 4.1 documented the GAT attention calculation





# 4.2 Single-head Implementation and Inspection

- Added src/models/gat.py and experiments/models/inspect_gat.py

    - Implemented a two-layer GAT graph classifier using standard GATConv layers

    - Used one head with 64 channels in each convolution as a teaching configuration

        - This precedes the reference multi-head architecture and is not the later fixed-total-width one-head experimental variant

    - The constructor accepted num_features, hidden_dim and num_classes

        - MUTAG supplied seven input features and two graph classes

        - hidden_dim=64 controlled both single-head convolution widths

- Connected the two attention blocks to the classifier

    - conv1 transformed seven input features into 64 features per node

    - F.elu applied the first block's elementwise output activation

    - conv2 received these 64-dimensional representations and produced another 64 features per node

        - Its coefficients were calculated from the first block's representations rather than directly from the original node features

    - A second F.elu followed conv2

    - global_add_pool(x, batch) summed node representations separately for each graph

    - nn.Linear(hidden_dim, num_classes) produced raw graph-class logits

        - No class softmax was applied inside forward

    - The convolutions had separate learned parameters, while each convolution shared its parameters across the nodes it processed

- Specified the GATConv dimensions and head combination

    - in_channels is the input feature count per node

    - out_channels is the output feature count per head

    - heads=1 configured one attention head in each convolution

    - conv1 used concat=True to concatenate head outputs

    - conv2 used concat=False to average head outputs

        - With one head, either operation retains that head's 64-dimensional output

        - Head combination does not average across neighbouring nodes; neighbourhood aggregation has already occurred within each head

- Specified scoring and dropout

    - negative_slope=0.2 controlled LeakyReLU inside attention scoring

    - dropout=0.0 disabled dropout on normalised attention coefficients

        - Coefficient dropout acts after neighbourhood softmax

        - Feature dropout instead masks coordinates of node representations

        - The supplied model used neither form of dropout

    - Zero dropout removes those sources of randomness but does not establish bit-for-bit deterministic execution

    - ELU remained an explicit operation after each convolution, separate from the scoring LeakyReLU

- Specified self loops, edge information, biases and residuals

    - add_self_loops=True included each node's own message

        - For ordinary edge_index tensors, the inspected published implementation removes existing self loops before inserting one per node

        - The model therefore left loop handling to the convolutions

    - edge_dim=None configured no edge-feature scoring transformation

        - forward passed only x and edge_index into each convolution

        - Available categorical bond labels were not consumed

        - fill_value concerns the edge features assigned to self loops and was irrelevant here

    - bias=True enabled an output bias after aggregation and head combination

        - The inspected shared feature transformation itself had no bias

    - residual=False disabled an additional learned skip connection

        - A self message weighted through attention is different from a separate residual path

- Mapped the named parameters to the single-head equation

    - lin.weight represented W, the shared node-feature transformation

    - att_src represented the sender scoring vector, and att_dst represented the receiver scoring vector

        - With i as receiver and j as sender, r_ij = att_dst^T z_i + att_src^T z_j

        - The two tensors together represented the two parts of the paper's concatenated scoring vector

    - The attention tensors had shape [1, heads, out_channels]

        - The leading singleton dimension allowed broadcasting across nodes

        - The middle dimension indexed heads

        - The final dimension contained each head's scoring-vector coordinates

    - bias represented the additive output bias

    - Attention coefficients did not appear in named_parameters()

        - They are calculated during forward execution from the learned parameters and the input graph

    - The inspected published implementation used Glorot initialisation for the feature transformation and scoring vectors, and zero initialisation for the convolution's output bias

        - Standard layer initialisation was retained without adding a custom initialiser

- Constructed and inspected a real MUTAG batch

    - TUDataset used root="data", name="MUTAG", cleaned=False, use_node_attr=False and use_edge_attr=False

        - These settings retained the established categorical node-feature input policy

        - The attribute flags do not remove all categorical edge labels; exclusion was enforced by the model inputs

    - DataLoader used batch_size=32 and shuffle=False

        - next(iter(loader)) selected the first batch in dataset order

    - Constructed the model using dataset.num_node_features, 64 and dataset.num_classes

    - model.eval() selected evaluation behaviour

        - It did not train the model or disable gradient recording

    - torch.no_grad() separately disabled gradient recording during inspection

    - model.named_parameters() supplied parameter names and tensors

        - parameter.shape exposed dimensions

        - parameter.numel() counted scalar entries

        - Component totals were calculated from conv1.parameters(), conv2.parameters() and classifier.parameters()

        - The trainable total included only parameters with requires_grad=True

    - Called the blocks separately to expose intermediate shapes, then called the complete model to exercise forward

        - This was an inspection of an untrained model, not a training or assessment run

- Executed python -m experiments.models.inspect_gat successfully

    - torch_geometric.__version__ printed 2.8.0.post1

    - The model printed GATConv(7, 64, heads=1), GATConv(64, 64, heads=1) and Linear(in_features=64, out_features=2, bias=True)

        - This abbreviated representation did not display every constructor option

        - The supplied source specified the remaining settings explicitly

- Reconciled the first convolution's parameters

    - conv1.lin.weight had shape [64, 7] and contained 448 parameters

        - The seven input features were mapped to 64 transformed features

    - conv1.att_src and conv1.att_dst each had shape [1, 1, 64] and contained 64 parameters

        - Together they supplied the 128 scoring coordinates needed for a pair of 64-dimensional transformed vectors

    - conv1.bias had shape [64] and contained 64 parameters

    - The first convolution contained 448 + 64 + 64 + 64 = 640 parameters

- Reconciled the second convolution and classifier parameters

    - conv2.lin.weight had shape [64, 64] and contained 4,096 parameters

        - Its input was the first block's 64-dimensional representation

    - conv2.att_src and conv2.att_dst each had shape [1, 1, 64] and contained 64 parameters

    - conv2.bias had shape [64] and contained 64 parameters

    - The second convolution contained 4,096 + 64 + 64 + 64 = 4,288 parameters

    - classifier.weight had shape [2, 64] with 128 parameters, and classifier.bias had shape [2] with two parameters

        - The classifier contained 130 parameters

    - Total parameters and trainable parameters both equalled 5,058

        - This matched 640 + 4,288 + 130

        - No model parameters were frozen

- Interpreted the observed tensor transitions

    - Input shape was [585, 7]

        - The batch contained 585 nodes across 32 graphs

        - Each node had seven input features

    - The first convolution produced [585, 64], and the first ELU retained that shape

    - The second convolution produced [585, 64], and the second ELU retained that shape

        - Message passing changed node representations while preserving one row per node

    - Sum pooling produced [32, 64]

        - Node rows were replaced by one summed representation per graph

    - The classifier produced [32, 2]

        - Each row contained two raw class scores for one graph

    - The separate complete model call also returned [32, 2]

        - This established successful execution of the supplied forward method with the expected output dimensions

- Recorded source provenance and verification limits

    - Consulted the official GATConv API and source

        - https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html

        - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html

    - The accessible published documentation described PyG 2.9.0, while the actual inspection ran with 2.8.0.post1

        - Execution established that the installed version accepted the supplied arguments and produced the recorded parameter and output shapes

        - The installed source was not independently inspected, so agreement on every internal implementation detail was not established

    - Attention coefficients were not extracted or checked for neighbourhood normalisation

    - The model remained untrained, so the inspection did not establish learned selectivity or predictive performance

- Commit: 4.2 added GAT and single-head inspection





# 4.3 Reference Multi-head Classifier

- Updated src/models/gat.py to the reference attention architecture

    - The first convolution used eight heads with 64 output channels each

        - concat=True combined their outputs into 512 features per node

    - The second convolution received 512 features and used one head to produce 64 features per node

    - Retained ELU after both convolutions, global sum pooling and a linear graph classifier

    - Retained the scoring, dropout, self-loop, edge-feature, bias and residual settings established in 4.2

    - The model remained a two-message-passing-layer classifier

- Explained what multiple heads share and what they learn separately

    - All first-layer heads received the same node features and operated over the same graph connectivity

        - Adding heads did not create different eligible neighbourhoods

    - Each head had its own feature transformation and sender/receiver scoring vectors

        - For head k, m_i^(k) = sum_{j in N_tilde(i)} alpha_ij^(k) W^(k) h_j

        - On MUTAG, each W^(k) had shape [64, 7]

        - Each head calculated and normalised its own coefficients over each receiving neighbourhood

    - Multiple heads therefore supplied separately parameterised aggregations in parallel

        - They could produce different representations and attention patterns

        - Separate parameters did not guarantee specialisation or prevent redundant learned behaviour

- Explained concatenation and the following layer

    - Concatenation placed the eight 64-dimensional head vectors alongside one another

        - The first block computed ELU([m_i^(1) || ... || m_i^(8)] + b_1)

        - || denoted concatenation and b_1 contained 512 coordinates

        - Eight heads multiplied by 64 channels produced 512 output features per node

    - Averaging eight 64-dimensional head outputs would instead retain 64 features

        - Concatenation preserves the head outputs as separate coordinate blocks

        - Averaging combines corresponding coordinates

    - The second convolution's transformation had shape [64, 512]

        - It could combine coordinates from all eight preceding heads

        - Its scores were calculated from the first block's complete representations

    - The second convolution used heads=1 and concat=False

        - concat=False averages the current convolution's own output heads

        - With one second-layer head, this left its output unchanged

        - It did not average the eight preceding first-layer heads

- Introduced explicit attention dimensions in the constructor

    - Replaced the teaching constructor's hidden_dim with attention_total_width, heads and embedding_dim

        - attention_total_width specified the concatenated first-layer width

        - heads specified the first-layer head count

        - embedding_dim specified the second-layer output and pooled graph-vector width

        - num_features and num_classes retained their dataset-dependent roles

    - The reference inspection supplied attention_total_width=512, heads=8 and embedding_dim=64

    - Derived channels_per_head using attention_total_width // heads

        - This gave 64 channels per head for the reference model

        - The constructor rejected heads below one and totals not exactly divisible by heads

        - This prevented integer division from silently producing an incompatible concatenated width

    - conv1 used channels_per_head as out_channels and heads as its head count

    - conv2 used attention_total_width as its input width, embedding_dim as its output width and one head

        - Its input dimension therefore came from the same value defining the first layer's total output

    - The classifier used nn.Linear(embedding_dim, num_classes)

    - forward retained the same convolution, activation, readout and classification sequence

- Established the later fixed-width head-study rule

    - Hold attention_total_width at 512 and embedding_dim at 64

        - One head requires 512 channels per head

        - Two heads require 256 channels per head

        - Four heads require 128 channels per head

        - Eight heads require 64 channels per head

    - The experimental factor changes how a fixed total representation width is divided among separate attention calculations

    - The earlier one-head, 64-channel teaching model is different from the later one-head, 512-channel experimental configuration

    - No head-study training runs were performed in this substage

- Updated experiments/models/inspect_gat.py

    - Constructed the reference model using explicit keyword arguments for the input features, total attention width, graph classes, first-layer heads and final embedding width

    - Printed conv1.heads and conv1.out_channels

        - These exposed the actual constructed head count and channels per head

        - Their product gave the total first-layer output width

    - Printed conv2.out_channels to identify the final node embedding width

    - Retained the named-parameter counts, intermediate shape inspection and complete model call

    - Retained the first non-shuffled MUTAG batch, model.eval() and torch.no_grad()

    - The model factory and training runner were not changed in this substage

- Executed python -m experiments.models.inspect_gat successfully

    - PyG printed version 2.8.0.post1

    - The configuration printed eight first-layer heads, 64 channels per head, total first-layer width 512 and final node embedding width 64

    - The model printed GATConv(7, 64, heads=8), GATConv(512, 64, heads=1) and Linear(in_features=64, out_features=2, bias=True)

        - The first convolution's printed 64 referred to channels per head, not its complete concatenated output width

- Reconciled the first convolution's parameters

    - conv1.lin.weight had shape [512, 7] and contained 3,584 parameters

        - PyG stored the eight [64, 7] head transformations together

        - The 512 output rows corresponded to eight blocks of 64 transformed features

    - conv1.att_src and conv1.att_dst each had shape [1, 8, 64] and contained 512 parameters

        - The middle dimension directly exposed the eight heads

        - Each head had 64 sender and 64 receiver scoring coordinates

    - conv1.bias had shape [512] and contained 512 parameters

        - It matched the concatenated output width

    - The first convolution contained 3,584 + 512 + 512 + 512 = 5,120 parameters

        - This was eight times the teaching first layer's 640 parameters because its per-head width remained 64

- Reconciled the second convolution and classifier parameters

    - conv2.lin.weight had shape [64, 512] and contained 32,768 parameters

        - Its larger input dimension accounted for most of the parameter increase over the teaching model

    - conv2.att_src and conv2.att_dst each had shape [1, 1, 64] and contained 64 parameters

    - conv2.bias had shape [64] and contained 64 parameters

    - The second convolution contained 32,768 + 64 + 64 + 64 = 32,960 parameters

    - classifier.weight remained [2, 64] with 128 parameters, and classifier.bias remained [2] with two parameters

        - The classifier retained its 130 parameters because the final graph-vector width remained 64

    - Total parameters and trainable parameters both equalled 38,210

        - This matched 5,120 + 32,960 + 130

        - The teaching model had contained 5,058 parameters

        - The increase involved both additional first-layer transformations and the larger second-layer transformation, not just additional scoring vectors

- Interpreted the observed tensor transitions

    - The input remained [585, 7], representing 585 nodes across 32 graphs

    - The first convolution produced [585, 512]

        - Eight 64-channel head outputs were concatenated for every node

        - The first ELU retained this shape

    - The second convolution produced [585, 64]

        - Its single head transformed the complete first-layer representation into 64 features

        - The second ELU retained this shape

    - Sum pooling produced [32, 64], and the classifier produced [32, 2]

        - The graph-level dimensions were unchanged from the teaching model

    - The separate complete model call also returned [32, 2]

        - This established forward execution through the reference architecture with the expected dimensions

- Derived the parameter-count consequence of fixed total width

    - Let T be total first-layer width and F be input feature count

    - With concatenation, the first convolution contains TF transformation parameters and T parameters for each of its sender scoring vector, receiver scoring vector and output bias

        - Its total is TF + 3T

        - This does not depend on how T is divided among heads

    - Holding T=512 and embedding_dim=64 also fixes the second convolution and classifier dimensions

    - The planned 1-, 2-, 4- and 8-head variants therefore each contain 38,210 parameters on MUTAG under this architecture

        - This equality was derived from the parameterisation

        - Only the eight-head reference configuration was executed in this substage

    - Equal parameter counts do not make the head configurations functionally identical or establish equal runtime

- Recorded architectural provenance and interpretation limits

    - Graph Attention Networks, Section 2.1, supplies the multi-head mechanism

        - The original paper motivates multiple heads as helping stabilise learning

        - That motivation does not establish a performance benefit for this untrained project model

    - The paper's Cora/Citeseer hidden configuration used eight heads with eight channels each

        - The project's eight-by-64 first layer is a deliberate width choice, not a reproduction of that configuration

    - The final 64-dimensional node embedding, ELU after both convolutions, sum readout and linear graph classifier form the project's graph-classification adaptation

    - The reference attention model has a wider intermediate representation than the 64-wide contextual baselines

        - Those comparisons do not isolate attention at equal capacity

        - The change from the teaching model also changes head count, total width and parameter count together

    - The observed output established construction, parameter counts and forward dimensions

        - Attention coefficients were not yet extracted

        - No model was trained or assessed for predictive performance

        - Stages 4.4 and 4.5 remained ahead

- Commit: 4.3 configured the reference multi-head GAT





# 4.4 Attention Coefficient Inspection

- Extended experiments/models/inspect_gat.py to extract attention from both reference GAT convolutions

    - Passed return_attention_weights=True when calling conv1 and conv2 during the shape inspection

    - Each call returned the node-output tensor together with a pair containing the effective edge_index and attention coefficients

        - The node output remained the convolution result before the external ELU

        - The returned edges described the messages used after self-loop handling

        - Each coefficient row corresponded to the same-position column of the returned edge_index

    - Stored these outputs as conv1_edge_index, conv1_attention, conv2_edge_index and conv2_attention

    - The second convolution received the first convolution's output after ELU, preserving the model's actual computation

    - Retained the complete model call after the intermediate inspection

        - Requesting attention coefficients added no learned parameters

- Corrected the inspector's inconsistent structure against the supplied current inspect_gin.py

    - The earlier GAT inspector used direct TUDataset construction, the variable batch, embedded newline characters and different common output labels

    - Replaced direct dataset construction with load_dataset("MUTAG") from src.data

        - Dataset-loading policy now came from the existing shared function

    - Used graph_batch consistently for the DataLoader batch

    - Used separate print() calls for blank lines and matched the existing multiline-call layout

    - Matched the common Model, Parameters, Shape transitions and Model output sections

    - Retained the common classifier and total parameter summaries

        - Removed the additional convolution totals, duplicate trainable total and repeated configuration/version output from this inspector

        - The previously recorded 4.2 and 4.3 observations remained valid historical results

    - Kept model-specific inspection where it had a scientific purpose

        - GIN's epsilon-buffer inspection remained specific to GIN

        - GAT's returned edges and coefficients formed its additional attention section

    - Used torch.nn.functional.elu for the GAT activations

        - This retained the intended ELU operation using the existing torch import

    - Retained model.eval() and torch.no_grad() for evaluation-mode attention inspection without gradient recording

- Defined how to interpret the returned tensors

    - For E' effective directed edge entries, attention_edge_index has shape [2, E']

        - Row 0 identifies sending nodes

        - Row 1 identifies receiving nodes

    - Attention weights have shape [E', heads]

        - Each row identifies an effective edge entry

        - Each column identifies one head

        - For an edge from j to i, column k contains alpha_ij^(k), using receiving-node-first mathematical notation

    - Used each convolution's own returned edges to interpret its coefficients

        - The original input edge_index cannot be assumed to retain the same count or ordering after self-loop handling

- Selected one receiving neighbourhood for readable inspection

    - Set receiving_node=0 before inspecting the coefficient values

        - This is node 0 in the batch, belonging to the first graph in the non-shuffled dataset batch

    - Formed incoming_mask = attention_edge_index[1] == receiving_node

        - The mask selected every effective edge entering that receiver, including its self loop

    - Printed attention_edge_index[:, incoming_mask].t()

        - Selecting columns retained the incoming edges

        - Transposing produced one [source, target] pair per printed row

    - Printed attention_weights[incoming_mask]

        - Applying the same mask preserved alignment between each displayed edge and its coefficient row

        - First-layer rows contained eight coefficients, while second-layer rows contained one

- Checked incoming-neighbourhood normalisation throughout the batch

    - Created incoming_sums using attention_weights.new_zeros((graph_batch.num_nodes, attention_weights.size(1)))

        - The accumulator had one row per node and one column per head

        - new_zeros retained the coefficient tensor's dtype and device

    - Used incoming_sums.index_add_(0, attention_edge_index[1], attention_weights)

        - Dimension 0 selected the accumulator's node rows

        - The receiving-node indices identified the destination row for each edge

        - Each edge's coefficient vector was added into that receiver's row

        - Head columns remained separate

        - The trailing underscore indicated an in-place update

    - Printed incoming_sums[receiving_node] for the selected example

    - Used torch.allclose against torch.ones_like(incoming_sums)

        - atol=1e-6 allowed small absolute floating-point differences

        - rtol=0.0 disabled additional relative tolerance

        - The check covered every receiving node and head in each convolution, not only the printed example

    - The required sum is over incoming messages within each head, not across head columns

    - Both layers used zero coefficient dropout, preserving the neighbourhood-softmax normalisation property during inspection

- Executed python -m experiments.models.inspect_gat successfully

    - The model retained GATConv(7, 64, heads=8), GATConv(512, 64, heads=1) and Linear(in_features=64, out_features=2, bias=True)

    - The classifier contained 130 parameters and the complete model contained 38,210 parameters

    - The shape progression remained [585, 7] to [585, 512] to [585, 64] to [32, 64] to [32, 2]

        - Both ELUs retained their respective input shapes

        - The complete model call also returned [32, 2]

- Interpreted the effective edge counts

    - The original edge_index had shape [2, 1304]

        - This represented 1,304 directed edge entries across the batch

    - Both convolutions returned edge_index with shape [2, 1889]

        - The increase was 1,889 - 1,304 = 585 entries

        - This matched the batch's 585 nodes and was consistent with adding one self loop per node without removing pre-existing loops

    - These counts describe directed message entries, not counts of distinct undirected bonds

- Interpreted first-layer attention

    - conv1_attention had shape [1889, 8]

        - Each effective edge had one coefficient from each of the eight heads

    - The incoming edges for receiving node 0 were [1, 0], [5, 0] and [0, 0], in that order

        - Nodes 1 and 5 supplied neighbour messages

        - Node 0 supplied its own message through the self loop

    - The corresponding attention tensor had three rows and eight columns

        - Every displayed coefficient was 0.3333

        - At the displayed precision, each head divided its weight equally among the three incoming messages

    - The printed incoming sums for node 0 were [1, 1, 1, 1, 1, 1, 1, 1]

    - The batch-wide normalisation check returned True

        - Every receiving-node/head sum was within the specified tolerance of one

- Interpreted second-layer attention

    - conv2_attention had shape [1889, 1]

        - Each effective edge had one coefficient from the second layer's single head

    - The selected incoming edges again appeared as [1, 0], [5, 0] and [0, 0]

    - Their displayed coefficients were 0.3333, 0.3333 and 0.3333

    - The selected node's incoming sum printed as [1]

    - The batch-wide normalisation check returned True

        - The second layer also satisfied the incoming-neighbourhood normalisation property within tolerance

- Interpreted the apparently uniform example cautiously

    - Uniform coefficients are a valid output of a learned attention mechanism

        - Equal logits among three eligible messages produce coefficients of 1/3

        - For a fixed receiver, identical sending-node representations produce equal logits within a head because the same transformation and scoring parameters are applied

    - The printed output did not include the relevant input or intermediate representations

        - Identical representations are therefore a possible explanation, not an established diagnosis of this example

    - The displayed coefficients were rounded

        - Printing 0.3333 for each entry does not establish exact numerical equality

        - The normalisation check established sums near one, not equality among individual coefficients

    - Inspecting one receiving neighbourhood did not establish uniform attention elsewhere in the graph or batch

    - The model was newly initialised and had not been trained

        - These coefficients illustrated implementation behaviour

        - They did not establish learned head redundancy, neighbour importance or predictive usefulness

        - This was not the later uniform-attention control, which requires explicitly constrained parameters and fresh training

- Recorded the agreed consistency work

    - Finish Stage 4 before beginning the wider repository audit

    - Audit current files one category at a time using the source supplied for that category

        - Compare scientific behaviour, shared structure, naming, formatting and explanatory writing

        - Where conventions differ and the preferred choice is not already settled, present the alternatives and establish the user's preference

        - Update affected logs where necessary while preserving actual execution history and recording subsequent corrections

    - Prefer short descriptive print labels with a few explanatory words

        - Avoid full-sentence labels

        - The long normalisation label in this execution remains part of the historical output; label standardisation is pending

- Commit: 4.4 added attention coefficient inspection