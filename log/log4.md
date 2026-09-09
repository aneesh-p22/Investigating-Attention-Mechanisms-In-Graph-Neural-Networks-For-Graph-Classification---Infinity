# 4.1 Attention Mathematics

- Worked through the single-head GAT calculation using Graph Attention Networks, Section 2.1, equations (1)-(4)

    - A GAT head transforms node features, calculates scalar scores for eligible source-receiver pairs, applies LeakyReLU, normalises the scores with neighbourhood softmax and aggregates the transformed vectors

    - The attention coefficients determine how strongly each eligible message contributes to the receiving node's update

    - Used deliberately chosen features and parameters for a mathematical example, not a trained model or an observed MUTAG result

- Defined the notation and dimensions

    - N is the number of nodes, F is the number of input features per node and F' is the number of output features per node for one head

    - i identifies the receiving node and j identifies the sending node, so alpha_ij weights the message travelling from j to i

    - h_j is the input representation of node j, written mathematically as a column vector with F coordinates

    - W is a learned matrix with shape [F', F], and z_j = W h_j is the transformed vector with F' coordinates

    - a_recv and a_send are learned scoring vectors, each with F' coordinates

    - b is an output bias with F' coordinates

    - F and F' describe feature dimensions, not node counts

- Established the graph and eligible incoming neighbourhoods

    - The three-node undirected graph connects node 1 to nodes 2 and 3, with no direct connection between nodes 2 and 3

    - Each undirected connection permits messages in both directions

    - Included one self loop per node and defined N_tilde(i) as the set of eligible sending nodes for receiver i, including i itself

    - The eligible sets are N_tilde(1) = {1, 2, 3}, N_tilde(2) = {1, 2} and N_tilde(3) = {1, 3}

    - The self loop allows the receiving node's own representation to participate through the same scoring and aggregation mechanism as its neighbours

    - Restricting attention to these eligible pairs is the graph masking operation; a node does not attend directly to every node in the graph

- Applied the shared feature transformation

    - Used F = F' = 2, with input vectors h_1 = [1, 0], h_2 = [0, 1] and h_3 = [1, 1], written horizontally here for readability

    - These illustrative features have no assigned chemical meaning

    - Chose W = [[1, 1], [0, 1]], giving z_1 = [1, 0], z_2 = [1, 1] and z_3 = [2, 1]

    - The first transformed coordinate adds the two input coordinates, while the second retains the second input coordinate

    - The same W is applied to every node within this head; its entries would be learned during training

    - The transformed vectors are used both to calculate attention scores and as the vectors being aggregated

- Calculated scalar pair scores

    - Original GAT uses r_ij = a^T [z_i || z_j], where || denotes concatenation and the superscript T denotes transpose

    - Concatenating the receiver and sender vectors produces 2F' coordinates, and the learned vector a has the same dimension

    - Splitting a into receiver and sender parts gives the equivalent expression r_ij = a_recv^T z_i + a_send^T z_j

    - Each dot product produces one scalar, and their sum is the pair score before LeakyReLU

    - Chose a_recv = [-2, 0] and a_send = [1, 1]

    - For receiver 1, a_recv^T z_1 = (-2)(1) + (0)(0) = -2

    - The sender contributions a_send^T z_j are 1, 2 and 3 for nodes 1, 2 and 3 respectively

    - The resulting scores are r_11 = -1, r_12 = 0 and r_13 = 1

    - These scores are not yet normalised weights

- Applied LeakyReLU to the pair scores

    - Defined L(t) = t for t >= 0 and L(t) = 0.2t for t < 0

    - The attention logits are e_ij = L(r_ij), giving e_11 = -0.2, e_12 = 0 and e_13 = 1

    - Negative scores remain negative but are reduced in magnitude

    - The negative branch has derivative 0.2, allowing a gradient to pass through that branch of the score calculation

    - LeakyReLU acts on scalar attention scores; ELU is a separate operation applied later to the aggregated node vector

- Normalised the logits with neighbourhood softmax

    - alpha_ij = exp(e_ij) / sum_{k in N_tilde(i)} exp(e_ik)

    - k runs over all eligible senders for the same receiver i, so each receiving neighbourhood has its own denominator

    - For receiver 1, the denominator is exp(-0.2) + exp(0) + exp(1)

    - The exponential values are approximately 0.818731, 1 and 2.718282, giving a denominator of approximately 4.537013

    - Dividing by this denominator gives alpha_11 = 0.180456, alpha_12 = 0.220409 and alpha_13 = 0.599135, approximately

    - The coefficients are positive and sum to one over the eligible incoming neighbourhood

    - These properties describe the softmax coefficients before coefficient dropout; the project uses zero coefficient dropout

    - Receiver 2 normalises over {1, 2} and excludes node 3, while receiver 3 normalises over {1, 3}

    - Although this is the same mathematical softmax operation used for class probabilities, these coefficients distribute weight across incoming messages rather than graph classes

- Aggregated the transformed vectors

    - Defined m_i = sum_{j in N_tilde(i)} alpha_ij z_j

    - Each alpha_ij is a scalar that multiplies the sender's entire F'-dimensional vector

    - For receiver 1, m_1 = 0.180456[1, 0] + 0.220409[1, 1] + 0.599135[2, 1], using rounded coefficients

    - Its first coordinate is approximately 0.180456 + 0.220409 + 2(0.599135) = 1.599135

    - Its second coordinate is approximately 0.220409 + 0.599135 = 0.819544

    - Therefore m_1 is approximately [1.599135, 0.819544]

    - Before adding a bias or applying an activation, this is a convex combination: a weighted sum with non-negative weights that sum to one

- Applied the output bias and node activation

    - The complete node update for the project's single-head block is h'_i = ELU(m_i + b)

    - Equivalently, h'_i = ELU(sum_{j in N_tilde(i)} alpha_ij W h_j + b)

    - The output bias is added after aggregation, and ELU acts separately on each coordinate of the resulting vector

    - With its scale parameter equal to one, ELU(t) = t for t > 0 and ELU(t) = exp(t) - 1 for t <= 0

    - Set b = [0, 0] for the numerical example

    - Both aggregated coordinates are positive, so ELU leaves them unchanged and h'_1 is approximately [1.599135, 0.819544]

    - All node updates in this layer use the same input-layer representations; updating node 2 does not use the newly calculated h'_1 during this same layer

- Connected the calculation to tensor storage

    - Mathematical node vectors were written as columns, while the node-feature tensor stores one node per row

    - H has shape [N, F], and the shared transformation is represented as Z = H W^T, producing shape [N, F']

    - Single-head aggregation and the elementwise activation retain shape [N, F']

    - The layer changes each node's representation while preserving one output row per node

    - Mapping these equations to actual GATConv parameters and inspecting the implementation remains the next substage

- Distinguished learned parameters from calculated coefficients

    - W, a_recv, a_send and b are learned parameters shared across nodes within a head

    - Connectivity and the self-loop convention determine which messages are eligible

    - Attention coefficients are recalculated from the current node representations, learned parameters and eligible neighbourhood

    - There is no independently stored trainable coefficient for every edge

    - During training, graph-classification gradients can propagate through aggregation, softmax and scoring to update the parameters

    - During evaluation, the learned parameters remain fixed, but different input representations or neighbourhoods can still produce different coefficients

- Compared attention with GCN normalisation

    - For an unweighted undirected graph, symmetric GCN normalisation assigns an eligible message the coefficient c_ij = 1 / sqrt(d_tilde_i d_tilde_j)

    - d_tilde_i is the degree of node i after adding self loops

    - The example degrees are d_tilde_1 = 3 and d_tilde_2 = d_tilde_3 = 2

    - Receiver 1 therefore has GCN coefficients 1/3, 1/sqrt(6) and 1/sqrt(6), approximately 0.333333, 0.408248 and 0.408248

    - GCN learns its feature transformation, while these normalisation coefficients are determined by connectivity

    - GAT additionally learns parameters that calculate coefficients from node representations

    - The example GAT assigns different coefficients to nodes 2 and 3 even though their degrees are equal

    - Symmetric GCN coefficients do not generally sum to one for a receiver

    - Equal GAT logits give alpha_ij = 1 / |N_tilde(i)| for each eligible sender in this simple graph, where |N_tilde(i)| is the number of eligible senders

    - For receiver 1, uniform GAT coefficients would therefore be 1/3, 1/3 and 1/3, producing a neighbourhood mean

    - Uniform GAT attention does not generally reproduce symmetric GCN normalisation

- Recorded the interpretation and execution boundary

    - Node 3 receives approximately 60% of receiver 1's incoming attention weight in this example

    - This describes its weight in the specified aggregation, not verified chemical importance or causal responsibility for a prediction

    - The calculation establishes the mathematical operation, not trained selectivity or improved graph-classification performance

    - No project source implementation or development fit was performed in this substage

- Commit: 4.1 documented the GAT attention calculation