# 5.1 Static Versus Dynamic Attention

- Studied the static-attention limitation of standard GAT before implementing GATv2

    - Used How Attentive Are Graph Attention Networks?, primarily Sections 3.1-3.3

    - The distinction concerns how the attention-scoring function can rank candidate sending nodes for different receiving nodes

    - Standard GAT can produce attention coefficients that vary between receivers

    - Its stronger restriction is that, within one attention head, changing the receiver cannot generally change the ordering of the same candidate senders

    - Brody, Alon and Yahav describe this property as static attention

    - GATv2 changes the order of operations in the attention-scoring function so that receiver-dependent rankings can be represented

        - The corresponding function family is capable of dynamic attention

- Recalled the standard GAT attention score

    - Let i denote the receiving node and j denote an eligible sending node

    - Let h_i and h_j denote their current node representations

    - Let W be the learned feature transformation used by one attention head

    - Define the transformed representations:

        - z_i = W h_i

        - z_j = W h_j

    - Standard GAT forms the unnormalised pair score:

        - r_ij = a^T [z_i || z_j]

    - The project then applies LeakyReLU:

        - e_ij = LeakyReLU(r_ij)

    - Neighbourhood softmax converts the resulting logits into attention coefficients:

        - alpha_ij = exp(e_ij) / sum_{k in N_tilde(i)} exp(e_ik)

    - N_tilde(i) contains the eligible sending nodes for receiver i, including the self connection used by the project

    - The static-ranking restriction arises from the pair-scoring expression before neighbourhood aggregation

- Split the standard GAT scoring vector into receiver and sender contributions

    - The attention vector a can be separated into two learned vectors:

        - a_recv for the receiver part

        - a_send for the sender part

    - The concatenated dot product can therefore be rewritten as:

        - r_ij = a_recv^T z_i + a_send^T z_j

    - Defined:

        - q_i = a_recv^T z_i

        - s_j = a_send^T z_j

    - The pair score becomes:

        - r_ij = q_i + s_j

    - q_i is a scalar determined by the receiving node

    - s_j is a scalar determined by the sending node

    - When receiver i compares several eligible senders, the same q_i is added to every sender score

- Derived why the sender ranking is static within one standard GAT head

    - Consider two candidate senders j and k that are both eligible for the same receiver

    - Suppose their sender contributions satisfy:

        - s_j > s_k

    - Their pre-activation scores are:

        - r_ij = q_i + s_j

        - r_ik = q_i + s_k

    - Adding the same q_i to both sides preserves the inequality:

        - q_i + s_j > q_i + s_k

    - Therefore:

        - r_ij > r_ik

    - LeakyReLU is strictly increasing

        - Positive inputs have slope one

        - Negative inputs have the project's slope 0.2

        - In either region, increasing the input increases the output

    - Applying LeakyReLU therefore preserves the ordering:

        - e_ij > e_ik

    - Neighbourhood softmax also preserves the ordering of logits for one receiver

        - Both coefficients use the same positive softmax denominator

        - The exponential function is strictly increasing

    - Therefore:

        - alpha_ij > alpha_ik

    - If sender j is ranked above sender k, changing the receiver contribution q_i cannot reverse that ordering

    - The same argument applies to any other receiver for which j and k are both eligible candidate senders

    - Standard GAT therefore cannot make one receiver prefer j over k while another receiver prefers k over j within the same head and for the same current representations

- Clarified what static attention does not mean

    - Static attention does not mean that every receiver receives identical attention coefficients

    - The receiver contribution q_i remains part of the standard GAT score

    - Changing q_i can change the numerical differences between the final LeakyReLU outputs

    - This can alter the resulting softmax coefficient values

    - The restriction concerns the ordering of candidate senders rather than requiring the complete coefficient vector to be identical

    - Static attention also does not mean that GAT ignores the receiving node

        - The receiver representation contributes to attention scoring through q_i

        - The receiver's neighbourhood determines which senders participate in the softmax

        - Its representation also changes between message-passing layers

    - The precise limitation is that one head cannot condition the relative ranking of shared candidate senders on the receiving representation

- Constructed an example showing that static ranking can still produce different coefficient values

    - Consider two candidate senders A and B

    - Chose their sender contributions as:

        - s_A = 2

        - s_B = 0

    - Therefore A has the larger sender contribution

    - Consider receiver 1 with:

        - q_1 = 1

    - The standard GAT pre-activation scores are:

        - r_1A = 3

        - r_1B = 1

    - Both values are positive, so LeakyReLU leaves them unchanged

    - The two-way softmax coefficients are approximately:

        - alpha_1A = 0.881

        - alpha_1B = 0.119

    - Sender A is ranked above sender B

    - Consider receiver 2 with:

        - q_2 = -1

    - Its pre-activation scores are:

        - r_2A = 1

        - r_2B = -1

    - With LeakyReLU slope 0.2, the logits become:

        - e_2A = 1

        - e_2B = -0.2

    - The two-way softmax coefficients are approximately:

        - alpha_2A = 0.769

        - alpha_2B = 0.231

    - The coefficient values changed between the two receivers

    - The ranking did not change:

        - A remained preferred to B for both receivers

    - This distinguishes static ranking from identical attention distributions

- Identified a stronger special case of the standard GAT restriction

    - Suppose all candidate pre-activation scores for a receiver lie on the same linear branch of LeakyReLU

    - On the positive branch:

        - e_ij = q_i + s_j

    - q_i is then a common additive shift applied to every candidate sender

    - Softmax is invariant to adding the same constant to every input

    - The receiver contribution therefore cancels completely from the resulting attention coefficients in this case

    - The same cancellation occurs when all compared values remain on the negative branch because the common receiver contribution is multiplied by the same LeakyReLU slope

    - Receiver-dependent coefficient changes can occur when candidate scores cross different LeakyReLU regions

    - Even then, the monotonicity of LeakyReLU prevents the sender ranking from reversing

- Connected the static-ranking result to multi-head GAT

    - A multi-head GAT layer contains separate learned attention parameters for each head

    - Different heads can therefore learn different sender-scoring functions

    - One head might rank sender A above sender B while another head ranks sender B above sender A

    - The static-attention argument nevertheless applies independently within each individual head

    - For a particular head, changing the receiver cannot reverse that head's ranking of the same candidate senders

    - Increasing the number of heads therefore provides multiple potentially different static rankings

    - It does not make the ranking inside an individual standard GAT head dynamically conditioned on the receiver

- Studied the paper's definitions of static and dynamic attention

    - Static attention describes a scoring function for which one key can remain highest-scoring across the considered queries

    - The preferred key is therefore not selected according to the individual query in the way required by a fully query-dependent ranking

    - Dynamic attention concerns a stronger attention function family

        - Over the considered finite query and key sets, it can represent arbitrary required query-to-key selections

        - Different queries can therefore require different keys to receive the highest score

    - The definitions concern the representational capability of attention-scoring functions

    - They do not state that every function representable by a dynamic attention family will actually learn a different ranking for every query

    - A model capable of dynamic attention can still learn behaviour that happens to look static on particular data

    - Static and dynamic are therefore theoretical properties that must be kept separate from observations of one trained model

- Examined how GATv2 changes the attention-scoring function

    - Standard GAT can be written in the compact form:

        - e_ij = LeakyReLU(a^T [W h_i || W h_j])

    - The concatenated receiver-sender representation is first reduced to one scalar by the dot product with a

    - LeakyReLU is then applied to that scalar

    - This ordering permits the score to be decomposed into the receiver scalar q_i and sender scalar s_j

    - GATv2 instead uses the compact scoring form:

        - e_ij = a^T LeakyReLU(W [h_i || h_j])

    - The paper equation is written here without the optional implementation bias terms

    - The important change is the order of the operations

        - The receiver and sender representation is transformed into a vector

        - LeakyReLU acts on the coordinates of that vector

        - The learned vector a then reduces the result to a scalar score

    - The nonlinear operation therefore occurs before the final scalar projection

    - The score can no longer generally be reduced to a monotone function of one receiver scalar plus one sender scalar

    - Changing the receiver can change which transformed coordinates are positive or negative

    - This can change the relative score assigned to different senders

- Constructed a small GATv2 example that permits a ranking reversal

    - Used one scalar query q and one scalar key k

    - Chose a two-coordinate transformed pair representation:

        - W[q || k] = [q - k, k - q]

    - Applied LeakyReLU separately to both coordinates

    - Chose the final scoring vector:

        - a = [-1, -1]

    - With LeakyReLU negative slope 0.2, the resulting score becomes:

        - e(q, k) = -LeakyReLU(q - k) - LeakyReLU(k - q)

    - This simplifies to:

        - e(q, k) = -1.2|q - k|

    - A key closer to the query therefore receives the larger score

    - Consider two keys:

        - k_0 = 0

        - k_2 = 2

    - For query q = 0:

        - e(0, 0) = 0

        - e(0, 2) = -2.4

        - k_0 receives the larger score

    - For query q = 2:

        - e(2, 0) = -2.4

        - e(2, 2) = 0

        - k_2 receives the larger score

    - Changing the query reversed the ranking of the same two candidate keys

    - This behaviour cannot be produced by the standard GAT ranking decomposition within one head

    - The example demonstrates the mechanism made possible by the GATv2 ordering of operations

    - It is an illustrative construction rather than a proof of the paper's full theorem

- Connected the example to the GATv2 expressiveness result

    - Brody, Alon and Yahav prove that the GATv2 scoring-function family is capable of dynamic attention over finite query and key sets

    - Their result uses the greater expressiveness introduced by applying the nonlinearity before the final scalar projection

    - The theorem establishes a representational capability

    - It does not guarantee that training will discover a dynamically varying ranking on a particular graph-classification dataset

    - It also does not guarantee that the additional capability will improve predictive accuracy

- Examined the DictionaryLookup experiment used by the paper

    - DictionaryLookup is designed so that different queries must select different corresponding keys

    - The correct highest-scoring key therefore depends directly on the query

    - This makes the task suitable for exposing the standard GAT ranking restriction

    - The paper reports that standard GAT struggles as the task becomes more demanding

    - GATv2 is able to solve the controlled task because its scoring formulation can represent the required query-dependent selections

    - The experiment provides an empirical example in which the theoretical distinction matters directly to the task

    - It does not establish that every real graph-classification problem requires dynamic attention

- Distinguished expressiveness from realised model behaviour

    - GATv2 being capable of dynamic attention does not mean that every trained GATv2 model will use dynamic rankings

    - A fitted GATv2 model could learn approximately the same sender ordering for many or all receivers

    - Observing such behaviour would not remove the model family's theoretical capability

    - Standard GAT can also perform well on a task even though its attention function is theoretically more restricted

    - A task may not require receiver-conditioned neighbour ranking

    - Other parts of the model can also contribute useful representations and predictive capacity

    - Theoretical attention expressiveness and graph-classification accuracy are therefore related but different questions

- Connected the theory to the project's second research question

    - RQ2 compares standard GAT with standard GATv2 under the project's matched graph-classification pipeline

    - The comparison asks whether the more expressive GATv2 attention formulation produces different realised graph-classification performance under the project's conditions

    - The shared comparison will retain the same overall graph-classifier structure

        - Two message-passing layers

        - Total hidden width 64

        - Eight heads in the first attention layer

        - One head in the second attention layer

        - ReLU after each complete convolution

        - Global sum pooling

        - Linear graph classifier

        - Shared training, validation and model-selection procedure

    - GATv2 will use its standard general-form implementation with separate receiver and sender transformations

    - The paper's reported experimental systems used shared transformations in relevant experiments

    - This implementation distinction must be recorded explicitly when GATv2 is introduced in Stage 5.2

    - The project will not alter the common hidden width merely to force the two models to have identical parameter counts

    - Their actual parameterisations and parameter counts will instead be inspected and reported

- Defined the interpretation boundary for RQ2

    - GATv2 has an attention-scoring capability that standard GAT lacks

    - This creates a theoretical reason why GATv2 could be advantageous when useful decisions require receiver-dependent neighbour ranking

    - It does not create a directional prediction that GATv2 must outperform GAT on MUTAG, PROTEINS or NCI1

    - If GATv2 performs better, the result alone cannot prove that dynamic ranking caused the improvement

        - The models also differ in their attention parameterisation

        - Optimisation behaviour can differ

        - Generalisation behaviour can differ

        - The datasets may use the available capacity in different ways

    - If GATv2 performs similarly to or worse than GAT, that result would not contradict the expressiveness theorem

        - Greater representational capacity does not require better performance on every task

    - RQ2 will therefore remain an empirical matched-pipeline comparison rather than being presented as a pure causal isolation of dynamic attention

- Recorded the scope of this substage

    - This substage established the theoretical motivation required before implementing GATv2

    - The standard GAT static-ranking restriction was derived directly from its scoring equation

    - Static sender ranking was distinguished from identical attention coefficients

    - The role of multi-head attention was kept separate from dynamic ranking within one head

    - The GATv2 ordering of operations was examined and a small example demonstrated receiver-dependent rank reversal

    - The paper's theoretical result was separated from claims about trained behaviour or graph-classification accuracy

    - RQ2 was framed as a matched empirical comparison with explicit interpretation limits

    - No GATv2 implementation or development fit was performed in this substage

    - Stage 5.2 will connect the GATv2 formulation to the installed GATv2Conv implementation, inspect its parameterisation and add the standard GATv2 model to the common training pipeline

- Commit: 5.1 documented static and dynamic graph attention