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