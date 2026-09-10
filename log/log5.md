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





# 5.2 Standard GATv2 and Development Fit

- Studied the GATv2 implementation before adding it to the common graph-classification pipeline

    - Used How Attentive Are Graph Attention Networks? together with the installed PyG GATv2Conv implementation

    - Kept the distinction between the general GATv2 formulation and the parameter-matched experimental restriction used in the paper

        - The paper's reported GATv2 experiments constrained the receiver and sender transformations to be shared

        - This was used to prevent an increased parameter count from explaining the reported GATv2 and GAT differences

        - The project instead uses the standard general-form GATv2 with separate receiver and sender transformations

        - This is represented by share_weights=False

    - The project therefore does not claim to reproduce the paper's parameter-matched experimental systems

    - RQ2 instead compares GAT and general-form GATv2 within the project's matched graph-classification pipeline while reporting their remaining parameterisation differences

- Inspected the installed GATv2Conv implementation

    - Confirmed the installed PyG version was 2.8.0.post1

    - Inspected the GATv2Conv constructor signature

    - Confirmed the relevant defaults and available options included:

        - heads=1

        - concat=True

        - negative_slope=0.2

        - dropout=0.0

        - add_self_loops=True

        - edge_dim=None

        - bias=True

        - share_weights=False

        - residual=False

    - Inspected the constructor implementation

        - lin_l transforms one side of the receiver-sender pair

        - lin_r transforms the other side

        - When share_weights=True, lin_r refers to the same transformation as lin_l

        - When share_weights=False, a separate lin_r transformation is constructed

        - Each transformation contains its own weight and, with bias=True, its own bias

        - The layer also contains a learned attention tensor and an output bias

    - Inspected edge_update

        - The transformed receiver and sender representations are added before attention scoring

        - LeakyReLU is applied to the resulting vector

        - The learned attention tensor is applied after this vector nonlinearity and summed across the output-channel dimension

        - Neighbourhood softmax then normalises the scalar scores for each receiving node

        - Attention dropout is applied after softmax, but the project uses dropout=0.0

    - This ordering matches the GATv2 mechanism studied in 5.1

        - Receiver and sender information interact before the final scalar attention projection

        - The standard GAT scalar decomposition responsible for static ranking therefore does not generally apply

- Implemented the project's standard GATv2 classifier

    - Added src/models/gatv2.py

    - Used two GATv2Conv layers followed by the same graph-classification structure used by the other models

    - The first layer takes the loaded node features and produces total width 64

        - Eight heads are used

        - Each head produces eight channels

        - Concatenating eight heads gives 64 output channels

    - The second layer takes 64 input channels and produces 64 output channels

        - It uses one head

        - concat=False retains the required final node-embedding width

    - Used share_weights=False in both GATv2 layers

    - Retained the shared attention settings:

        - LeakyReLU negative slope 0.2 inside attention scoring

        - Zero attention-coefficient dropout

        - Self connections enabled

        - No edge-feature scoring

        - Bias enabled

        - No residual transformation

    - Applied ReLU after each complete convolution

    - Used global sum pooling and the same linear graph classifier

    - The model returns raw graph logits

- Connected GATv2 to the shared model construction and development settings

    - Added GATv2 to the common model builder

    - Kept heads and share_weights explicit rather than hiding the scientific configuration inside the model

    - Added the GATv2 model settings:

        - heads=8

        - share_weights=False

    - Retained the common development settings

        - hidden_dim=64

        - learning rate 0.01

        - weight decay 0.0005

        - batch size 32

        - 1000 epochs

        - training seed 0

        - split seed 0

    - No GATv2-specific hyperparameter search or development-score tuning was introduced

- Added and ran the GATv2 model inspector

    - Constructed GATv2 on MUTAG before training

    - The model contained:

        - GATv2Conv(7, 8, heads=8)

        - GATv2Conv(64, 64, heads=1)

        - Linear(64, 2)

    - Inspected every learned parameter tensor

    - The first convolution contained:

        - att with shape [1, 8, 8] and 64 parameters

        - output bias with shape [64] and 64 parameters

        - lin_l weight with shape [64, 7] and 448 parameters

        - lin_l bias with shape [64] and 64 parameters

        - lin_r weight with shape [64, 7] and 448 parameters

        - lin_r bias with shape [64] and 64 parameters

    - The first convolution therefore contained 1152 parameters

    - The second convolution contained:

        - att with shape [1, 1, 64] and 64 parameters

        - output bias with shape [64] and 64 parameters

        - lin_l weight with shape [64, 64] and 4096 parameters

        - lin_l bias with shape [64] and 64 parameters

        - lin_r weight with shape [64, 64] and 4096 parameters

        - lin_r bias with shape [64] and 64 parameters

    - The second convolution therefore contained 8448 parameters

    - The classifier contained 130 parameters

    - The complete model therefore contained 9730 parameters

    - Confirmed share_weights was False in both convolutions

    - Confirmed lin_l and lin_r were distinct transformation objects in both convolutions

    - The larger parameter count relative to GAT is therefore an actual consequence of the selected separate-transform GATv2 parameterisation

- Inspected the GATv2 tensor shapes through the graph-classification pipeline

    - The inspected batch contained 585 nodes from 32 MUTAG graphs

    - Input node features had shape [585, 7]

    - The first convolution produced [585, 64]

    - The first ReLU retained [585, 64]

    - The second convolution produced [585, 64]

    - The second ReLU retained [585, 64]

    - Global sum pooling produced [32, 64]

    - The classifier produced [32, 2]

    - The complete model forward pass also returned [32, 2]

    - This confirmed that GATv2 preserves the common width, readout and classifier interface expected by the shared training code

- Inspected the untrained attention output

    - The original batch edge index had shape [2, 1304]

    - Both GATv2 layers returned an edge index with shape [2, 1889]

    - The increase of 585 entries matched the 585 nodes in the batch

        - This was consistent with one effective self connection being included for each node

    - The first layer returned attention coefficients with shape [1889, 8]

        - One coefficient was returned for every effective edge entry and each of the eight heads

    - The second layer returned attention coefficients with shape [1889, 1]

        - One coefficient was returned for every effective edge entry because the layer has one head

    - Inspected receiver node 0

        - Its eligible incoming entries were from senders 1, 5 and itself

    - The untrained first-layer coefficients were approximately one third for all three entries in every displayed head

    - The untrained second-layer coefficients were also approximately one third for the three entries

    - The incoming coefficients summed to one separately for every head

    - This established the expected neighbourhood softmax normalisation for the inspected example

    - The equal values were an observation from this fresh untrained model and were not interpreted as a general property of GATv2 or as evidence about learned attention

- Committed the implemented GATv2 source before recording its development fit

    - This ensured the result could record the source state that actually produced the training run

    - The recorded source commit was 513d6d1c6630eec657db5241a63191a4703f41dd

    - This avoids fitting from scientifically relevant uncommitted source changes

- Recorded the representative GATv2 development fit on MUTAG

    - Used the existing stratified development partition

        - 150 training graphs

        - 18 validation graphs

        - 20 development-test graphs

    - Used CUDA

    - Trained for all 1000 fixed epochs

        - No early stopping was used

        - Validation was evaluated after every epoch

        - Console progress was printed every ten epochs

    - Selected the model state using strict minimum validation cross-entropy

    - The selected state occurred at epoch 823

        - Epoch 823 was not displayed in the ten-epoch progress output because model selection still operated on every epoch

    - The selected validation loss was 0.2357

    - The selected validation accuracy was 0.8333

    - Some other epochs had higher validation accuracy

        - This does not affect checkpoint selection because validation loss, rather than validation accuracy, is the predefined selection criterion

    - Restored the selected state before development-test evaluation

    - Development-test loss was 0.4311

    - Development-test accuracy was 0.8000

    - The complete model contained 9730 total parameters

    - All 9730 parameters were trainable

    - Measured training time was 46.5741 seconds

    - Mean measured training time was 0.0466 seconds per epoch

    - Saved the selected model state to results/development_gatv2_mutag_seed0.pt

    - Saved the paired result record to results/development_gatv2_mutag_seed0.json

- Interpreted the development result within its intended scope

    - The fit establishes that general-form GATv2 can be constructed, trained, validation-selected, restored, evaluated and persistently saved through the same project pipeline as the existing models

    - The development result is implementation evidence rather than final RQ2 assessment evidence

    - The earlier development GAT and current GATv2 runs can provide contextual checks but do not form the final controlled comparison

    - In these development runs, both models obtained development-test accuracy 0.8000

    - GATv2 had selected validation loss 0.2357 compared with the earlier GAT value 0.2423

    - GATv2 had selected validation accuracy 0.8333 compared with the earlier GAT value 0.9444

    - GATv2 had development-test loss 0.4311 compared with the earlier GAT value 0.4058

    - These mixed observations give no justified development-stage claim that either attention formulation is superior

    - The final RQ2 comparison will require the predefined final evaluation procedure across the complete adopted dataset set

    - GATv2's greater theoretical attention expressiveness does not guarantee higher graph-classification accuracy

    - The GAT and GATv2 models also do not have equal parameter counts under the project's general-form share_weights=False choice

        - This remaining capacity difference must be disclosed when interpreting RQ2

        - The common representation width will not be distorted merely to force parameter equality

- Recorded the scope of this substage

    - Standard general-form GATv2 is now part of the common graph-classification pipeline

    - Its installed scoring implementation, transformation sharing behaviour, biases and parameter count were explicitly inspected

    - Its common tensor interface and local attention normalisation were checked before training

    - One validation-selected MUTAG development state and result record were preserved

    - No final RQ2 conclusion was drawn from the development fit

    - No additional trained-neighbourhood GATv2 inspector was added because this substage did not require another local trained example

    - Stage 5.3 will define the minimal core ablation variants without creating a general experiment framework

- Commit: 5.2 added GATv2 and recorded its development fit





# 5.3 Minimal Ablation Runner

- Defined the minimal set of core attention variants before building the final cross-validation runner

    - Added experiments/ablation.py

    - Kept the file deliberately small rather than creating a general experiment framework

    - The runner currently defines and inspects the configurations that will be used for the three confirmed attention research questions

    - No model fitting or assessment-data evaluation is performed by this substage

    - Final cross-validation execution will be connected only after the shared final evaluation machinery exists in Stage 6

- Reused the existing shared settings rather than duplicating the training configuration

    - Imported settings and model_settings from experiments/train.py

    - Importing the file does not start a development run because its execution is protected by the main guard

    - Copied the common settings into base_settings

    - This preserves the existing hidden width, optimiser settings, epoch count, batch size and seed configuration

    - Variant dictionaries then override only settings that genuinely differ for the corresponding comparison

    - This avoids creating a second independently editable copy of the common experimental configuration

- Defined the RQ1 fixed-width head-count variants

    - Added standard GAT configurations with one, two, four and eight first-layer heads

    - Kept hidden_dim=64 for every configuration

    - The GAT constructor derives the number of channels per head by dividing hidden_dim by heads

    - The inspected one-head configuration produced:

        - 64 channels per head

        - First-layer width 64

        - Final embedding width 64

    - The inspected two-head configuration produced:

        - 32 channels per head

        - First-layer width 64

        - Final embedding width 64

    - The inspected four-head configuration produced:

        - 16 channels per head

        - First-layer width 64

        - Final embedding width 64

    - The inspected eight-head configuration produced:

        - Eight channels per head

        - First-layer width 64

        - Final embedding width 64

    - RQ1 therefore varies the partition of a fixed 64-dimensional first-layer representation across attention heads

    - It does not vary the total first-layer representation width

    - The second GAT layer remains one 64-channel head through the existing model definition

- Assigned explicit variant identities to the additional RQ1 fits

    - The one-head GAT uses variant heads1

    - The two-head GAT uses variant heads2

    - The four-head GAT uses variant heads4

    - These are marked as additional CV fits

    - The ordinary eight-head GAT keeps variant=None

    - The eight-head configuration is marked for reuse of the standard reference GAT CV results

    - This prevents a scientifically redundant second eight-head GAT fit under an ablation-specific filename

- Defined reuse of the standard GATv2 reference for RQ2

    - Added the standard GATv2 configuration to the core variant definitions

    - Retained heads=8

    - Retained share_weights=False

    - Retained variant=None

    - Marked the configuration for reuse of the standard GATv2 reference CV results

    - RQ2 will therefore compare the ordinary reference GAT and ordinary reference GATv2 results rather than create duplicate ablation runs

- Predeclared the RQ3 uniform-attention condition

    - Added a GAT variant named uniform

    - Added uniform_attention=True to its scientific settings

    - Retained the standard eight-head GAT width configuration

        - Eight heads

        - Eight channels per head

        - First-layer width 64

        - Final embedding width 64

    - Marked the uniform condition as requiring an additional CV fit

    - The uniform_attention setting currently identifies the intended experimental condition only

    - Stage 5.4 will implement and verify the actual uniform-attention constraint before this variant can be trained

- Kept execution metadata separate from scientific model settings

    - Each variant records whether it requires an additional CV fit or can reuse a reference result

    - This is stored as runner metadata through reuse_reference

    - reuse_reference is not inserted into the effective scientific settings dictionary

    - It therefore does not become part of the model configuration or result settings

    - In contrast, uniform_attention is part of the scientific settings because it will change how the uniform GAT is constructed and trained

- Inspected the current result-path identities

    - Used the shared get_result_path function rather than reproducing filename construction inside the ablation runner

    - The one-head configuration currently maps to results/final_gat_mutag_heads1_seed0.json

    - The two-head configuration currently maps to results/final_gat_mutag_heads2_seed0.json

    - The four-head configuration currently maps to results/final_gat_mutag_heads4_seed0.json

    - The eight-head reference currently maps to results/final_gat_mutag_seed0.json

    - The GATv2 reference currently maps to results/final_gatv2_mutag_seed0.json

    - The uniform GAT currently maps to results/final_gat_mutag_uniform_seed0.json

    - The standard reference models correctly contain no unnecessary variant suffix

    - The additional ablation conditions are distinguished by their variant suffixes

- Recorded the current limitation of the displayed final paths

    - The existing result recorder currently represents purpose, model, dataset, variant and seed

    - Final cross-validation fold identities have not yet been implemented

    - The printed final paths are therefore inspections of the current model and variant naming flow rather than the locked Stage 7 assessment filenames

    - Stage 6 will extend the common final-result machinery with the required fold information

    - The ablation runner will use that shared machinery rather than implement an independent naming system

- Recorded the scope of this substage

    - The confirmed core attention comparisons now have explicit machine-readable configurations

    - RQ1 has fixed-width one, two, four and eight-head GAT configurations

    - The standard eight-head GAT result is designated for reuse rather than duplicate training

    - The standard GATv2 result is designated for reuse in RQ2

    - The uniform GAT has been predeclared as a separate RQ3 training condition

    - No general experiment framework was introduced

    - No development or final model was trained

    - No assessment result was inspected

    - Stage 5.4 will implement and verify the uniform-attention GAT condition

- Commit: 5.3 added the minimal core ablation runner





# 5.4 Establish Uniform-Attention Implementation

- Defined the uniform-attention condition required for RQ3

    - RQ3 compares standard learned GAT attention against a GAT retrained from scratch with structurally uniform neighbourhood coefficients

    - The uniform condition retains the standard GAT graph-classification architecture

    - It does not modify a fitted reference GAT after training

    - The uniform model will later receive its own training run under the same final cross-validation protocol

    - The intervention therefore removes learned attention selectivity during training rather than applying a post-training coefficient replacement

- Derived how standard GAT can be constrained to uniform attention

    - Standard GAT uses learned source and destination attention-scoring tensors

    - PyG stores these as att_src and att_dst

    - Setting both scoring contributions to zero makes every eligible receiver-sender attention score equal before softmax

    - For a receiver with d eligible incoming entries, softmax over equal logits gives coefficient 1/d to every entry

    - This property applies separately within each attention head

    - Zero attention-coefficient dropout preserves the normalised uniform coefficients used by the project

- Distinguished zero initialisation from a permanent uniform-attention constraint

    - Initialising the attention-scoring tensors to zero would not be sufficient by itself

    - If the tensors remained trainable, gradient updates could immediately move them away from zero

    - The uniform condition therefore both zeros and freezes att_src and att_dst in both GAT layers

    - Frozen scoring tensors remain part of the model but cannot be changed by optimisation

    - Other GAT parameters remain trainable

        - Feature-transformation weights remain learned

        - Convolution biases remain learned

        - The graph classifier remains learned

    - The intervention therefore removes learned neighbour selectivity without removing representation learning from the model

- Extended the existing GAT model with an explicit uniform-attention option

    - Added uniform_attention=False to the GAT constructor

    - The default False value preserves the existing standard GAT behaviour

    - When uniform_attention=True:

        - conv1.att_src is set to zero

        - conv1.att_dst is set to zero

        - conv2.att_src is set to zero

        - conv2.att_dst is set to zero

        - All four tensors have requires_grad disabled

    - Used torch.no_grad while setting the parameter values to zero

    - The existing two-layer GAT architecture otherwise remains unchanged

        - First-layer total width 64

        - Eight heads for the standard uniform condition

        - Second layer with one 64-channel head

        - ReLU after each complete convolution

        - Global sum pooling

        - Linear graph classifier

        - Self connections enabled

        - Zero attention-coefficient dropout

        - No edge-feature scoring

        - No residual transformation

- Connected the uniform condition to the common model builder

    - The GAT builder now reads uniform_attention from the effective settings

    - Used False when the setting is absent

    - Existing standard GAT configurations therefore remain valid without adding a new setting to every historical configuration

    - The uniform variant declared in 5.3 already supplies uniform_attention=True

    - That variant can now construct the constrained GAT rather than serving only as an experimental label

- Added a compact uniform-GAT scientific inspector

    - Added experiments/models/inspect_uniform_gat.py

    - Constructed the model through the common model builder

    - Used MUTAG, hidden width 64 and eight first-layer heads

    - Inspected the four attention-scoring tensors directly

    - Checked each tensor's shape, parameter count, trainability and values

    - Checked total, trainable and frozen parameter counts

    - Constructed the parameter collection intended for optimisation using only parameters with requires_grad=True

    - Inspected actual attention coefficients from both GAT layers for one predetermined receiving node

    - Compared those coefficients with the analytically expected uniform value

- Verified the frozen attention-scoring parameters

    - conv1.att_src had shape [1, 8, 8]

        - It contained 64 parameters

        - It was non-trainable

        - Every value was zero

    - conv1.att_dst had shape [1, 8, 8]

        - It contained 64 parameters

        - It was non-trainable

        - Every value was zero

    - conv2.att_src had shape [1, 1, 64]

        - It contained 64 parameters

        - It was non-trainable

        - Every value was zero

    - conv2.att_dst had shape [1, 1, 64]

        - It contained 64 parameters

        - It was non-trainable

        - Every value was zero

    - The four frozen scoring tensors therefore contained 256 parameters in total

- Verified the parameter accounting

    - The complete uniform GAT retained 5058 total parameters

    - This matches the total parameter count of the standard eight-head GAT architecture

    - 4802 parameters remained trainable

    - 256 parameters were frozen

    - The difference between total and trainable parameters exactly matched the four frozen attention-scoring tensors

    - The filtered parameter collection intended for the optimiser contained 4802 parameters

    - The uniform intervention therefore freezes only the intended attention-scoring parameters rather than removing or freezing unrelated GAT parameters

- Updated optimiser construction to use only trainable parameters

    - The shared training runner now filters model parameters using requires_grad before passing them to Adam

    - Standard models are unaffected because their ordinary parameters remain trainable

    - The uniform GAT excludes its four frozen attention-scoring tensors from the optimiser

    - This keeps optimiser participation consistent with the scientific definition of the intervention

- Verified the resulting uniform attention coefficients

    - Inspected receiver node 0

    - Its incoming entries were from senders 1, 5 and itself

    - The receiving neighbourhood therefore contained three eligible entries

    - The analytically expected uniform coefficient was 1/3

        - This is approximately 0.333333

    - In the first convolution:

        - Each of the eight heads assigned approximately 0.3333 to sender 1

        - Each head assigned approximately 0.3333 to sender 5

        - Each head assigned approximately 0.3333 to the self entry

        - The three incoming coefficients summed to one separately for every head

        - The explicit uniformity check returned True

    - In the second convolution:

        - The single head assigned approximately 0.3333 to each of the three incoming entries

        - The coefficients summed to one

        - The explicit uniformity check returned True

- Distinguished the uniform result from the untrained attention observations in earlier inspections

    - Fresh untrained attention models can sometimes display equal or nearly equal coefficients for a particular neighbourhood

    - Such an observation alone does not imply that their attention is structurally constrained to remain uniform

    - In the RQ3 uniform model, the scoring tensors are permanently zero and frozen

    - Equal logits are therefore guaranteed by the implemented constraint for every eligible neighbourhood

    - For a receiver with d eligible entries, every head produces coefficient 1/d before any coefficient dropout

    - The project uses zero coefficient dropout

    - Uniformity is therefore a model property here rather than an accidental observation of one initial parameter state

- Distinguished uniform GAT from GCN

    - Uniform GAT does not become a GCN when its learned attention scores are removed

    - Uniform GAT applies equal neighbourhood softmax weights to its transformed messages

    - These weights form a neighbourhood mean for each receiving node

    - GCN instead uses symmetric degree-normalisation coefficients derived from graph connectivity

    - GCN coefficients do not generally sum to one for each receiving neighbourhood

    - The models also retain their own parameterisations and update equations

    - RQ3 therefore tests learned attention selectivity against uniform GAT aggregation rather than comparing GAT against GCN under another name

- Defined the interpretation boundary for the future RQ3 experiment

    - The uniform model will be retrained from scratch under the same final protocol as standard GAT

    - A performance difference will therefore compare models that had the opportunity to adapt their remaining trainable parameters to their respective attention mechanisms

    - The experiment does not measure the immediate effect of replacing coefficients inside one already-fitted GAT

    - That would be a different fitted-model intervention question

    - The confirmed RQ3 comparison should therefore be interpreted as learned-attention retraining versus structurally uniform-attention retraining

- Verified persistence of the uniform-attention constraint through an optimiser update

    - Performed one ordinary forward pass, cross-entropy backward pass and Adam update on the disposable inspection model

    - The optimiser received only the 4802 parameters whose requires_grad value was True

    - After the update, conv1.att_src remained unchanged and entirely zero

    - conv1.att_dst remained unchanged and entirely zero

    - conv2.att_src remained unchanged and entirely zero

    - conv2.att_dst remained unchanged and entirely zero

    - The first message-transformation weight changed during the same update

    - The classifier weight also changed during the same update

    - The probe therefore established that training can update the intended representation and classification parameters while the attention-scoring mechanism remains permanently fixed to uniform weighting

- Recorded the scope of this substage

    - The uniform-attention variant declared in 5.3 now has a concrete model implementation

    - Its four attention-scoring tensors are zero and permanently frozen

    - Its total and trainable parameter counts were verified directly

    - Its optimiser parameter set was verified to contain only trainable parameters

    - Actual PyG attention outputs confirmed the analytically expected uniform coefficients in both layers

    - The shared optimiser construction now filters out frozen parameters

    - No uniform development fit was performed

    - No final RQ3 assessment result was produced or inspected

    - Stage 5.5 will lock the research design and source notes needed before moving into the final experimental protocol

- Commit: 5.4 established the uniform-attention GAT





# 5.5 Research Design and Source Notes

- Defined the umbrella research question for the attention investigation

    - The project asks what aspects of graph attention design affect graph-classification behaviour under controlled evaluation

    - Predictive accuracy is the primary behaviour used to answer the confirmed questions

    - Runtime provides practical computational context

    - The design does not assume in advance that attention must outperform the contextual models

    - It also does not assume that more attention heads, GATv2 or learned non-uniform coefficients must produce higher accuracy

    - Negative, mixed and dataset-dependent outcomes remain valid evidence

- Locked RQ1 as the fixed-width GAT head-count comparison

    - RQ1 asks how the number of GAT attention heads affects graph-classification performance when total first-layer representation width is fixed

    - The changed scientific factor is the number of first-layer attention heads

        - One head

        - Two heads

        - Four heads

        - Eight heads

    - hidden_dim remains 64 for every head configuration

    - The derived channels per head are therefore:

        - 64 channels for one head

        - 32 channels for two heads

        - 16 channels for four heads

        - Eight channels for eight heads

    - Concatenation therefore produces first-layer width 64 in every condition

    - The second layer remains one 64-channel head

    - The eight-head condition is the ordinary reference GAT and is reused rather than retrained under another variant identity

    - The original GAT paper motivates multi-head attention as beneficial for stabilising self-attention learning

    - This provides a reason to study head organisation but does not imply that accuracy must increase monotonically with head count

    - Holding total width fixed also creates a trade-off

        - Increasing the number of heads creates more independently parameterised attention mechanisms

        - Each individual head receives fewer output channels

    - Non-monotonic or dataset-dependent behaviour is therefore plausible

    - RQ1 cannot establish a universal optimal number of heads

    - It also cannot establish what happens when both head count and total representation width increase together

- Locked the fixed factors for RQ1

    - The same dataset and outer-fold partition will be used across compared head configurations

    - Both message-passing layers retain the common project structure

    - Total hidden and final node-embedding width remains 64

    - ReLU remains the external activation after each complete convolution

    - Attention-score LeakyReLU retains negative slope 0.2

    - Self connections remain enabled

    - Attention-coefficient dropout remains zero

    - Edge features remain excluded

    - Residual transformations remain disabled

    - Global sum pooling and the linear graph classifier remain unchanged

    - The same categorical node-input policy is used within each dataset

    - Adam, learning rate 0.01, weight decay 0.0005, batch size 32 and exactly 1000 epochs remain fixed

    - No early stopping or head-specific hyperparameter search is introduced

    - The same validation-state selection and final fold procedure will be used for all conditions

- Locked RQ2 as the standard GAT and standard general-form GATv2 comparison

    - RQ2 asks how standard GAT compares with standard GATv2 under the matched graph-classification pipeline

    - The standard reference GAT and GATv2 final results will be reused directly

    - Duplicate ablation-specific reference fits will not be created

    - The theoretical motivation comes from the different attention-scoring function families

        - Standard GAT computes only static attention within an individual head for fixed node representations

        - Its sender ranking cannot generally be conditioned on the receiving query

        - GATv2 changes the order of the nonlinear scoring operations

        - Its scoring-function family can represent dynamic query-dependent rankings

    - The GATv2 expressiveness result provides a reason to investigate possible empirical differences

    - It does not imply a directional graph-classification accuracy guarantee

- Recorded the controlled factors and remaining difference for RQ2

    - GAT and GATv2 use the same two-layer graph-classifier structure

    - Both use first-layer total width 64

    - Both use eight first-layer heads of eight channels

    - Both use one 64-channel second-layer head

    - Both use ReLU after each complete convolution

    - Both use the same node inputs and graph connectivity

    - Both use global sum pooling and the same linear classifier structure

    - Both use the same optimiser settings, epoch budget, folds and validation-state selection

    - The project GATv2 uses share_weights=False in both layers

    - It therefore uses separate receiver and sender transformations

    - This is the standard general-form choice adopted by the project

    - It differs from the shared-transform restriction used in the reported experiments of How Attentive Are Graph Attention Networks?

    - The paper imposed shared transformations to rule out an increased parameter count as the explanation for empirical differences

    - Its Appendix G parameter analysis distinguishes the unconstrained GATv2 parameterisation from the shared experimental form

    - The installed project GATv2 was measured at 9730 parameters on MUTAG

    - The standard eight-head GAT contains 5058 parameters on MUTAG

    - Equal hidden widths therefore do not imply equal model capacity

    - The project will report this difference rather than distort representation widths to force parameter equality

- Defined the interpretation boundary for RQ2

    - Better GATv2 accuracy would be consistent with the possibility that its more expressive attention formulation is useful for the task

    - Such a result would not isolate query-dependent ranking as the sole cause

    - The standard formulations also differ in parameterisation and parameter count

    - Optimisation and generalisation behaviour can also differ

    - Similar or worse GATv2 accuracy would not contradict its theoretical dynamic-attention capability

    - Greater function-family expressiveness does not require better performance on every dataset

    - The project therefore treats RQ2 as a matched standard-formulation comparison rather than a pure causal ablation

- Locked RQ3 as learned GAT attention against uniform-attention retraining

    - RQ3 asks whether learning non-uniform GAT neighbourhood coefficients improves performance relative to uniform-attention retraining

    - The reference condition is the ordinary eight-head GAT

    - The comparison condition is a fresh eight-head GAT trained with uniform coefficients in both layers

    - The uniform model fixes att_src and att_dst to zero in both layers

    - The four scoring tensors remain frozen throughout optimisation

    - Equal logits therefore produce coefficient 1/d for each of a receiver's d eligible incoming entries

    - The model is retrained from scratch with this constraint rather than created by modifying a fitted reference GAT

- Recorded the literature motivation for RQ3

    - Graph Attention Networks reported a Const-GAT control with the same architecture and constant attention

    - The control assigned the same importance to each neighbour

    - The paper used it to evaluate the benefit of the learned attention mechanism in its PPI experiment

    - Learned GAT outperformed the constant-attention control in that source experiment

    - This provides direct motivation for testing learned versus constant weighting

    - The source experiment was a different dataset, task and architecture setting

    - Its positive direction is therefore not assumed to transfer to this project's graph-classification datasets

- Locked the fixed factors and necessary difference for RQ3

    - Standard and uniform GAT retain the same two-layer architecture and total width

    - Both retain the same head structure

    - Both retain the same feature transformations, biases, ReLU activations, sum readout and classifier structure

    - Both receive the same inputs, graph partitions and training procedure

    - Standard GAT learns its attention-scoring tensors

    - Uniform GAT retains those tensors in the architecture but fixes them permanently to zero

    - Standard eight-head GAT contains 5058 total and trainable parameters on MUTAG

    - Uniform GAT also contains 5058 total parameters

    - Freezing the four 64-parameter scoring tensors leaves 4802 trainable parameters

    - This trainable-parameter difference is an intended consequence of removing learned attention scoring

    - It will be reported rather than hidden

- Defined the interpretation boundary for RQ3

    - A standard GAT advantage would provide evidence that allowing learned non-uniform weighting was useful under the tested retraining conditions

    - A uniform-GAT advantage or similar result would show that learned unequal weighting was not necessary for better performance under those conditions

    - Neither outcome establishes that attention is universally necessary or unnecessary

    - RQ3 does not measure what happens when coefficients are changed inside an already fitted learned GAT

    - A fitted-model intervention is a separate possible investigation

- Defined the core three-dataset experiment matrix before final assessment

    - MUTAG, PROTEINS and NCI1 remain the confirmed datasets

    - Each dataset will receive the same core model and variant matrix

    - The contextual reference configurations are:

        - GCN

        - GraphSAGE

        - GIN

    - The attention configurations are:

        - GAT with one first-layer head

        - GAT with two first-layer heads

        - GAT with four first-layer heads

        - Reference GAT with eight first-layer heads

        - Reference GATv2 with eight first-layer heads and share_weights=False

        - Uniform-attention GAT with eight first-layer heads

    - The reference eight-head GAT serves simultaneously as:

        - The eight-head RQ1 condition

        - The GAT side of RQ2

        - The learned-attention side of RQ3

    - The reference GATv2 result serves directly as the GATv2 side of RQ2

    - Reusing these reference groups avoids scientifically redundant duplicate fits

    - The current matrix therefore contains nine unique configurations for each dataset

    - Across three datasets this produces 27 dataset-configuration groups

    - Ten final folds per group imply 270 core model fits

    - These counts follow from the predeclared matrix and do not depend on observed model rankings

- Recorded the common evidence plan for the confirmed questions

    - Final assessment will use the same stratified outer-fold procedure for every compared condition

    - Configurations are fixed before final outcomes are viewed

    - Accuracy is the primary predictive measure

    - Runtime is retained as practical context rather than replacing predictive assessment

    - Fold-level results will be preserved rather than reporting only one aggregate number

    - Comparisons will retain negative and inconsistent fold or dataset behaviour

    - The eventual summaries will distinguish descriptive performance evidence from stronger causal or universal claims

- Recorded source anchors for the attention research design

    - Graph Attention Networks, Section 2.1 motivates multiple independent attention heads and reports them as beneficial for stabilising self-attention learning

    - The same paper reports a Const-GAT experiment using an identical architecture with constant neighbour importance

    - That constant control provides literature precedent for asking whether learned unequal weighting is useful

    - How Attentive Are Graph Attention Networks?, Section 3 defines static and dynamic attention

    - Its Theorem 1 establishes the static-ranking restriction of standard GAT

    - Its multi-head discussion states that this restriction applies independently to each standard GAT head

    - Its Theorem 2 and Appendix A establish the dynamic-attention capability of the GATv2 scoring-function family

    - Its Section 4 setup constrains the receiver and sender transformations to be shared in the reported GATv2 experiments

    - Appendix G.2 explains that this restriction removes the extra transformation parameters so increased parameter count cannot explain the paper's comparison

    - The project deliberately uses the general separate-transform GATv2 instead and records that distinction

    - The paper's discussion also states that a theoretically stronger mechanism need not be the practically best model for every task

    - These sources motivate the questions without supplying predetermined answers for this project's datasets

- Kept dataset-wide learned-attention characterisation as an unadopted candidate investigation

    - Candidate investigation A would characterise learned attention over a predefined collection of graphs rather than rely on one attractive local example

    - A scientifically clean option is to use preserved validation-selected final states and predefined held-out graph coverage

    - Out-of-fold coverage could allow every graph to be characterised using a model for which that graph was held out

    - Candidate measurements to consider include:

        - Attention concentration or entropy

        - Departure from uniform weighting

        - Self-connection attention mass

        - Similarity between attention heads

    - These are candidate measurements rather than adopted metrics

    - At 6.3 the exact question must be defined before deciding which measurements are necessary

    - The decision must also define which model or models are analysed

    - Receiver, head, layer, graph and fold aggregation must be specified before assessment

    - The interpretation would concern learned coefficient behaviour rather than automatically proving feature importance, prediction causality or chemical significance

- Kept fitted-model attention intervention as a separate unadopted candidate investigation

    - Candidate investigation B would ask how a fitted model's predictions respond to a predefined change in its learned attention

    - A minimal possible design would preserve a selected trained GAT and compare its ordinary predictions with predictions from a copied state whose attention scoring is replaced by a predefined alternative

    - Uniform scoring in both layers is one concrete intervention to assess at 6.3

    - The remaining learned message transformations and classifier would not be retrained during this intervention

    - The same held-out graphs could therefore provide paired prediction evidence before and after intervention

    - Before adoption, the intervention must define:

        - Exactly which scoring parameters or coefficients are changed

        - Which layers and heads are affected

        - How coefficient normalisation is preserved

        - Which predictive endpoints are compared

        - Which controls prevent unrelated model changes

    - This candidate is distinct from RQ3

        - RQ3 retrains a fresh model under permanently uniform attention

        - The candidate intervention would modify attention inside an already fitted learned model without retraining its remaining parameters

    - Such an intervention could measure model sensitivity to its fitted attention mechanism

    - It would not by itself establish chemical, biological or real-world causal importance

- Kept COLLAB as an independent optional dataset decision for Stage 6.3

    - COLLAB remains the most likely fourth-dataset candidate but has not been adopted

    - It will not be added merely to increase the number of datasets

    - Before adoption, Stage 6.3 must verify its original source and actual loader behaviour

    - Its available node and edge information must be inspected before defining a compatible project input policy

    - Computational cost must be evaluated using the development measurements available by that point

    - If adopted into the complete core matrix, COLLAB would add nine configuration groups

    - Ten folds for those groups would add 90 model fits

    - The resulting four-dataset core matrix would contain 360 fits

    - The dataset decision is independent of whether either candidate attention analysis is adopted

- Preserved the distinction between confirmed and conditional research scope

    - RQ1, RQ2 and RQ3 are confirmed questions

    - Their comparison conditions have been defined before final assessment outcomes

    - Dataset-wide attention characterisation is not currently RQ4

    - Fitted-model attention intervention is not currently RQ5

    - COLLAB is not currently part of the active dataset matrix

    - Stage 6.3 can adopt neither, either or both candidate analyses and can decide the dataset independently

    - Additional question numbers and exact metrics will be assigned only after an explicit adoption decision

    - No final ranking will be used to decide whether a candidate investigation or dataset is included

- Recorded the scope of this substage

    - The three confirmed attention questions now have explicit motivations, changed factors, fixed factors and interpretation limits

    - Their core model and dataset matrix has been predeclared

    - Reference-result reuse has been established before final outcomes

    - Original GAT constant-attention evidence and GATv2's shared-versus-general parameterisation distinction have been preserved as source notes

    - The two possible additional attention investigations have been converted into concrete decision options without being presented as adopted work

    - COLLAB remains an independent decision for Stage 6.3

    - No Background, Introduction or Methodology report prose was drafted

    - No final assessment result was generated or inspected

    - Stage 5 research design is now ready for closing notes and the Stage 5 review before Stage 6 begins

- Commit: 5.5 recorded the core attention experiment design





# Stage 5 Closing Notes

- Decisions

    - Adopted standard general-form GATv2 as the project's second principal attention model

        - Used share_weights=False in both GATv2 layers

        - Retained separate receiver and sender transformations rather than the shared-transform restriction used in the paper's reported experiments

        - Kept this distinction explicit because it changes the model parameterisation and prevents RQ2 from being interpreted as a pure parameter-matched causal comparison

    - Matched GATv2 to the reference GAT in the common graph-classification structure

        - Two message-passing layers

        - First-layer total width 64

        - Eight first-layer heads of eight channels

        - One 64-channel second-layer head

        - ReLU after each complete convolution

        - Global sum pooling

        - Linear graph classifier

        - Shared node inputs, graph connectivity and training protocol

    - Kept the actual GAT and GATv2 parameter-count difference rather than changing representation widths to force equality

        - Reference GAT contains 5058 parameters on MUTAG

        - General-form GATv2 contains 9730 parameters on MUTAG

        - This remaining capacity difference will be disclosed when interpreting RQ2

    - Locked RQ1 as a fixed-total-width GAT head-count comparison

        - The tested first-layer head counts are 1, 2, 4 and 8

        - Total first-layer width remains 64

        - Channels per head are therefore 64, 32, 16 and 8 respectively

        - The ordinary eight-head reference GAT is reused as the eight-head RQ1 condition

    - Locked RQ2 as the matched-pipeline comparison between standard GAT and standard general-form GATv2

        - The ordinary reference GAT and GATv2 results will be reused directly

        - Duplicate ablation-specific reference fits will not be created

        - The comparison will not be described as isolating dynamic attention alone because parameterisation, optimisation and generalisation can also differ

    - Locked RQ3 as standard learned GAT attention against uniform-attention retraining

        - The uniform model is trained from scratch under a permanent attention constraint

        - att_src and att_dst are set to zero and frozen in both GAT layers

        - All remaining representation and classifier parameters remain trainable

        - The optimiser receives only parameters whose requires_grad value is True

        - The intervention therefore removes learned neighbour selectivity without converting the model into GCN

    - Established the uniform-attention control before final assessment

        - The constrained GAT retains 5058 total parameters

        - Freezing the four 64-parameter attention-scoring tensors leaves 4802 trainable parameters

        - Equal attention logits produce coefficient 1/d for each of a receiver's d eligible incoming entries

        - The constraint was verified through an actual optimiser update rather than only at initialisation

    - Defined a minimal explicit ablation configuration instead of a general experiment framework

        - Additional fits are defined only for GAT heads 1, 2 and 4 and uniform GAT

        - Reference eight-head GAT and reference GATv2 results are marked for reuse

        - Variant construction derives the shared defaults from the existing common settings

        - Final cross-validation execution will be connected after the common CV driver exists

    - Predeclared the core three-dataset matrix before final assessment outcomes

        - MUTAG, PROTEINS and NCI1 remain the confirmed datasets

        - Each dataset has nine unique model or variant configurations

        - The complete three-dataset core therefore contains 27 dataset-configuration groups

        - Ten outer folds imply 270 core fits

    - Kept COLLAB outside the active matrix until the explicit Stage 6.3 decision

        - Its source, loader behaviour, usable input policy and computational cost must be established before adoption

        - If adopted into the complete core matrix, it would add 90 fits and increase the core total to 360

    - Kept the possible dataset-wide attention analysis and fitted-model intervention separate from the three confirmed questions

        - Neither has been promoted to RQ4 or RQ5

        - Their exact questions, measurements, controls and evidence requirements must be decided before adoption

        - Their inclusion will not depend on whether the final reference results produce an attractive model ranking

- Ideas

    - Assess at Stage 6.3 whether a dataset-wide learned-attention characterisation would add enough interpretive value to justify inclusion

        - A preferred design would use predefined graph coverage and preserved selected states rather than selected illustrative examples

        - Possible measurements include attention concentration, departure from uniform weighting, self-connection mass and similarity between heads

        - Exact measurements remain undecided until the scientific question is fixed

    - Assess independently at Stage 6.3 whether a fitted-model attention intervention would provide a useful additional question

        - A possible design would compare a preserved fitted GAT with the same state after a predefined attention intervention without retraining the remaining parameters

        - This would answer a different question from uniform-attention retraining and must remain clearly separated from RQ3

    - Assess COLLAB independently from the two candidate attention analyses

        - The project may adopt neither, one or both candidate analyses regardless of whether COLLAB is added

- Report notes

    - The installed PyG version inspected for GATv2 was 2.8.0.post1

    - Installed GATv2Conv exposes share_weights=False as its default

    - With share_weights=False, lin_l and lin_r are separate learned transformations

    - The installed GATv2 edge-scoring implementation adds the transformed receiver and sender representations, applies vector-valued LeakyReLU, applies the learned attention tensor and then performs neighbourhood softmax

    - This implementation corresponds to the GATv2 ordering studied in the static-versus-dynamic attention analysis

    - The GATv2 paper's shared-transform experimental restriction must remain distinct from the project's general separate-transform implementation

    - Standard GAT's static-attention restriction concerns the ranking of shared candidate senders within an individual head

        - It does not imply that every receiver has identical coefficient values

        - It does not imply that the receiving representation is absent from the model

        - Different GAT heads can learn different static rankings

    - GATv2 has a theoretically stronger attention-scoring function family capable of dynamic query-dependent rankings

        - This is an expressiveness result rather than a guarantee of higher graph-classification accuracy

        - A GATv2 advantage would not independently prove that dynamic ranking caused the difference

        - Similar or worse GATv2 performance would not contradict the expressiveness result

    - The GATv2 MUTAG development fit was produced from source commit 513d6d1c6630eec657db5241a63191a4703f41dd

    - The GATv2 development fit completed all 1000 epochs and selected epoch 823 by minimum validation cross-entropy

    - Selected validation loss was 0.2357

    - Selected validation accuracy was 0.8333

    - Development-test loss was 0.4311

    - Development-test accuracy was 0.8000

    - The fitted GATv2 contained 9730 total and trainable parameters

    - Measured training time was 46.5741 seconds

    - Mean measured training time was approximately 0.0466 seconds per epoch under the established timing convention

    - The selected GATv2 state was saved to results/development_gatv2_mutag_seed0.pt

    - Its paired result was saved to results/development_gatv2_mutag_seed0.json

    - The GATv2 development result is implementation evidence rather than final RQ2 assessment evidence

        - It had slightly lower selected validation loss than the earlier GAT development fit

        - It had lower selected validation accuracy

        - Both development fits obtained development-test accuracy 0.8000

        - GATv2 had somewhat higher development-test loss

        - These mixed single-split observations do not justify a model-ranking conclusion

    - The uniform GAT inspection confirmed 5058 total, 4802 trainable and 256 frozen parameters

    - All four frozen scoring tensors were exactly zero before training

    - For the inspected three-entry neighbourhood, every first-layer head and the second-layer head produced approximately one-third attention coefficients

    - Each inspected head's incoming coefficients summed to one

    - After one Adam update, all four attention-scoring tensors remained unchanged and exactly zero

    - During that same update, the first message-transformation weight and classifier weight changed

    - The probe therefore demonstrated persistence of the uniform constraint while retaining ordinary learning in the intended parts of the model

    - Uniform GAT remains mathematically distinct from GCN

        - Uniform GAT uses equal neighbourhood-softmax coefficients

        - GCN uses symmetric degree normalisation

    - The original GAT constant-attention experiment provides literature precedent for testing whether learned unequal neighbour weighting is beneficial

    - The GATv2 static and dynamic attention results provide the theoretical motivation for RQ2 without supplying a predetermined empirical answer

    - Final conclusions must retain the distinction between theoretical expressiveness, learned coefficient behaviour, predictive performance and causal interpretation

- Stage 5 recorded GATv2, controlled attention variants and the predeclared research design