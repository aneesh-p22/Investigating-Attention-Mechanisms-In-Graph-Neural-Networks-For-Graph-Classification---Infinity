# 6.1 PROTEINS Inspection and Development Pass

- Added experiments/datasets/inspect_proteins.py to inspect the second core graph-classification dataset before using it for a recorded development fit.

    - load_dataset("PROTEINS") loaded the dataset through the existing common TUDataset path and therefore retained the same project feature policy used for MUTAG.

    - dataset.num_node_features reported the width of the node-feature matrix actually supplied to the models.

    - dataset.num_node_labels and dataset.num_node_attributes distinguished categorical node-label channels from separately available continuous node attributes.

    - getattr(graph, "edge_attr", None) was used because graphs can legitimately have no loaded edge-feature tensor. PROTEINS returned None rather than an empty feature matrix.

    - unique(dim=0) was used across the loaded node-feature matrices to inspect the distinct categorical feature rows without printing an arbitrary collection of individual nodes.

- The command python -m experiments.datasets.inspect_proteins loaded PROTEINS successfully and reported 1,113 graphs in two processed classes.

    - Class 0 contained 663 graphs and class 1 contained 450 graphs.

    - The biological task was recorded as enzyme versus non-enzyme protein classification, but no biological meaning was assigned to numeric class IDs 0 and 1 because that numeric mapping was not established by the inspection.

- PROTEINS provides both categorical and continuous node information, but only the categorical information is part of the settled model input.

    - Loaded node-feature width was 3.

    - Node label channels were 3.

    - One continuous node attribute was separately available.

    - The loaded categorical rows were [1, 0, 0], [0, 1, 0] and [0, 0, 1]. Each loaded node therefore belongs to one of three categorical node types represented as a three-dimensional one-hot vector.

    - The continuous node attribute was deliberately excluded under the existing use_node_attr=False policy. Its existence was inspected so that this exclusion is explicit rather than mistaken for missing data.

    - The purpose of retaining the three categorical channels while excluding the continuous attribute is to keep the previously defined input policy fixed across models. Introducing the continuous attribute during the attention investigation would change the information available to every model as an additional experimental factor.

- PROTEINS had no edge-feature channels in the loaded representation.

    - Loaded edge features, edge label channels and continuous edge attributes were all reported as zero.

    - edge_attr was None for both representative graphs and for the inspected minibatch.

    - The models therefore receive graph connectivity through edge_index but no edge features on PROTEINS.

- One representative graph from each processed target class was inspected.

    - Dataset index 0 had target tensor([0]), 42 nodes, x shape [42, 3] and edge_index shape [2, 162].

    - Dataset index 663 had target tensor([1]), 32 nodes, x shape [32, 3] and edge_index shape [2, 128].

    - The first dimension of edge_index remained 2 because each stored connection entry is represented by a source-node index and a target-node index. The second dimension gives the number of stored edge entries for that graph.

    - x used torch.float32 because its one-hot values are model input features, while edge_index and y used torch.int64 because they contain indices and class IDs respectively.

- PROTEINS contains substantial variation in graph size.

    - Graphs contained between 4 and 620 nodes, with a mean of 39.06 nodes.

    - Stored edge entries ranged from 10 to 2,098, with a mean of 145.63.

    - Every graph had node-feature width 3 and edge-feature width 0.

    - These are stored edge-entry counts from the PyG representation. They were not relabelled as numbers of unique undirected relationships.

- A deterministic 32-graph DataLoader batch established how the larger PROTEINS graphs enter the unchanged graph-classification pipeline.

    - The batch contained 2,404 nodes, so x had shape [2404, 3].

    - edge_index had shape [2, 9364].

    - y had shape [32], giving one graph target for each graph in the minibatch.

    - batch had shape [2404] and contained graph IDs 0 through 31. Each entry assigns one concatenated node row to its graph within the minibatch.

    - ptr contained 33 cumulative node boundaries for the 32 graphs. For example, its first values 0, 42 and 69 show that graph 0 occupies the first 42 node rows and graph 1 occupies the following 27 rows.

    - All 32 targets in this inspection batch were class 0 because the inspection DataLoader used shuffle=False and these early dataset entries belong to class 0. The training DataLoader uses shuffle=True, so this ordered inspection batch does not describe training minibatch composition.

- The existing stratified_split procedure was applied with split seed 0.

    - Training contained 890 graphs with class counts [530, 360].

    - Validation contained 111 graphs with class counts [66, 45].

    - Development test contained 112 graphs with class counts [67, 45].

    - The split is the established approximately 80/10/10 development allocation performed within each class. Integer allocation within the two classes accounts for the final 890/111/112 graph totals.

- experiments/train.py was configured for one ordinary GCN development fit on PROTEINS while leaving the established training protocol unchanged.

    - hidden_dim remained 64.

    - Adam used learning rate 0.01 and weight decay 0.0005.

    - Training used 1,000 epochs, batch size 32, training seed 0 and split seed 0.

    - Validation was evaluated after every epoch. The selected state was the earliest state obtaining the strict minimum graph-mean validation cross-entropy, while training still continued for all 1,000 epochs.

    - The selected state was restored after training and was the state evaluated once on the development-test partition and saved to disk.

- The recorded fit was produced from source commit 677ddb0812e0081d85e6db60f244338caf1a1050 and ran on CUDA.

    - The selected state occurred at epoch 297.

    - Its validation loss was 0.4909 and its validation accuracy was 0.7748.

    - Epoch 297 does not appear in the ten-epoch progress output because progress printing and state selection have different frequencies. Selection evaluates every epoch.

    - The printed epoch-1000 validation loss was 0.5387, compared with the selected minimum of 0.4909. This illustrates why the fixed protocol restores the validation-selected state rather than using the final training epoch automatically.

- The restored selected state obtained development-test loss 0.5715 and development-test accuracy 0.7411.

    - This result establishes a working ordinary GCN development path on PROTEINS.

    - It is development evidence rather than final cross-validation evidence and is not used to tune the shared architecture or optimisation settings.

- The PROTEINS GCN contained 4,546 parameters, all of which were trainable.

    - The first GCN layer maps 3 input channels to 64 outputs, giving 3 × 64 = 192 weight parameters and 64 bias parameters, for 256 parameters.

    - The second GCN layer has 64 × 64 = 4,096 weight parameters and 64 bias parameters, for 4,160 parameters.

    - The graph classifier has 64 × 2 = 128 weight parameters and 2 bias parameters, for 130 parameters.

    - The total is 256 + 4,160 + 130 = 4,546.

    - MUTAG used seven rather than three input channels, so the PROTEINS GCN has 4 × 64 = 256 fewer first-layer weights while retaining the same hidden width and remaining architecture.

- The measured training-pass time was 148.2629 seconds, with a mean of 0.1483 seconds per epoch.

    - This timing covers the established training-pass operations rather than complete program wall-clock time. Validation, development-test evaluation, selected-state copying and restoration, progress printing, state saving and result writing are excluded by the recorded runtime convention.

    - The measurement is retained as practical evidence for the later Stage 6 run-budget decision rather than as a basis for altering the model.

- The restored selected state was saved as results/development_gcn_proteins_seed0.pt and its settings, partitions, selection evidence, development-test metrics, parameter counts, runtime, device and source provenance were saved as results/development_gcn_proteins_seed0.json.

- Commit: 6.1 inspected PROTEINS and recorded its development fit





# 6.2 NCI1 Inspection and Development Pass

- Added experiments/datasets/inspect_nci1.py to inspect the third core graph-classification dataset before its recorded development fit.

    - load_dataset("NCI1") used the existing common TUDataset loading path, preserving the same cleaned=False, use_node_attr=False and use_edge_attr=False dataset policy already used for MUTAG and PROTEINS.

    - dataset.num_node_features reported the width of the node-feature vectors actually supplied to the models, while dataset.num_node_labels and dataset.num_node_attributes distinguished categorical node-label channels from separately available continuous attributes.

    - NCI1 has a much larger categorical node vocabulary than the earlier datasets, so the inspection checked the structure of the complete vocabulary rather than assigning unsupported meanings to individual channels.

    - argmax(dim=1) identified the active position in each one-hot node-feature row. unique() then retained each occurring channel index once and sort() placed those indices in numerical order.

    - A dataset-wide one-hot check separately verified that every entry was either 0 or 1 and that every node-feature row summed to exactly 1. This distinguishes a genuinely one-hot categorical representation from merely observing vectors that happen to have width 37.

- The command python -m experiments.datasets.inspect_nci1 loaded NCI1 successfully and reported 4,110 graphs in two processed target classes.

    - Class 0 contained 2,053 graphs and class 1 contained 2,057 graphs, so the processed dataset is almost exactly balanced between its two classes.

    - The graph task was recorded as chemical-compound activity classification for non-small-cell lung cancer screening.

    - Nodes were represented as categorically labelled atoms and edges represented molecular connectivity.

    - No active or inactive biological meaning was assigned to numeric class IDs 0 and 1 because the inspection established the processed numeric targets but not their semantic mapping.

- NCI1 loaded 37 categorical node-feature channels and no continuous node attributes.

    - The complete dataset contained 37 distinct node-feature rows.

    - Every loaded row passed the one-hot check.

    - The active channel indices covered every integer from 0 through 36, confirming that all 37 categorical channels occur somewhere in NCI1.

    - A 37-channel vocabulary does not mean that every graph contains all 37 categories. Each node occupies one active channel, and individual graphs may use only a subset of the dataset-wide vocabulary.

    - The categorical indices are specific to NCI1's encoding. Their chemical identities were not inferred from the channel meanings established previously for MUTAG.

- NCI1 contained no loaded edge-feature information.

    - Loaded edge features, edge label channels and continuous edge attributes were all reported as zero.

    - edge_attr was None for both representative graphs and for the inspected minibatch.

    - The principal models therefore use molecular connectivity through edge_index, with no edge-feature tensor available to the model on this dataset.

- One representative graph from each processed target class was inspected.

    - Dataset index 0 had target tensor([0]), 21 nodes, x shape [21, 37] and edge_index shape [2, 42].

    - Its nodes used categorical channels 0, 1 and 2.

    - Dataset index 1650 had target tensor([1]), 36 nodes, x shape [36, 37] and edge_index shape [2, 76].

    - Its nodes used categorical channels 0, 1, 2 and 4.

    - These examples show how the shared 37-dimensional vocabulary is used sparsely within individual molecular graphs while retaining one common input dimension for the model.

- Dataset-wide graph sizes were also inspected because they affect batching and computational cost.

    - Graphs contained between 3 and 111 nodes, with a mean of 29.87 nodes.

    - Stored edge entries ranged from 4 to 238, with a mean of 64.60.

    - Every graph had node-feature width 37 and edge-feature width 0.

    - These values are counts from the PyG representation. Stored edge entries were not reinterpreted as a count of unique undirected chemical bonds.

- A deterministic DataLoader batch of 32 graphs contained 707 nodes.

    - x had shape [707, 37], so the node rows from all 32 graphs were concatenated while retaining the 37-dimensional NCI1 feature representation.

    - edge_index had shape [2, 1502].

    - y had shape [32], providing one target for every graph in the batch.

    - batch had shape [707] and contained graph IDs 0 through 31, assigning each concatenated node row to its original graph within the minibatch.

    - ptr contained 33 cumulative node boundaries. Its opening values 0, 21, 45 and 74 show that the first graph contributes 21 nodes, the second contributes 24 and the third contributes 29.

    - All 32 graph targets in this inspected minibatch were class 0 because shuffle=False preserved the ordering of the first dataset entries. The training loader uses shuffle=True, so this deterministic inspection batch is not the training batch distribution.

- stratified_split was applied with the established split seed 0.

    - Training contained 3,287 graphs with class counts [1642, 1645].

    - Validation contained 410 graphs with class counts [205, 205].

    - Development test contained 413 graphs with class counts [206, 207].

    - The near-equal class counts in each partition follow from the near-balanced complete dataset and the existing class-wise approximately 80/10/10 allocation.

- experiments/train.py was configured for one ordinary GCN development fit on NCI1.

    - Only the selected dataset changed from the preceding PROTEINS development fit.

    - hidden_dim remained 64.

    - Adam retained learning rate 0.01 and weight decay 0.0005.

    - The fixed budget remained 1,000 epochs with batch size 32, training seed 0 and split seed 0.

    - Validation continued to select the earliest state attaining the strict minimum graph-mean validation cross-entropy after each epoch, without early stopping.

- The recorded fit ran on CUDA from source commit 8b0203ab11ddc92a893b786cf4de38070eca841d.

    - The selected state occurred at epoch 6.

    - Its validation loss was 0.6116 and its validation accuracy was 0.6780.

    - Progress is printed only every ten epochs, while validation selection occurs after every epoch. The selected epoch therefore occurs before the first displayed training-progress line without indicating any inconsistency.

    - Training still completed all 1,000 epochs. The early selected state records the minimum validation-loss state rather than terminating optimisation.

- The restored selected state obtained development-test loss 0.5912 and development-test accuracy 0.6901.

    - This is development evidence showing that the existing graph-classification and selected-state pipeline operates correctly on NCI1.

    - The result is not final cross-validation evidence and was not used to revise the shared architecture, optimiser or training settings.

- The NCI1 GCN contained 6,722 parameters, all of which were trainable.

    - The first GCN layer maps 37 input channels to 64 outputs, giving 37 × 64 = 2,368 weight parameters and 64 bias parameters, for 2,432 parameters.

    - The second GCN layer contains 64 × 64 = 4,096 weight parameters and 64 bias parameters, for 4,160 parameters.

    - The graph classifier contains 64 × 2 = 128 weight parameters and 2 bias parameters, for 130 parameters.

    - The total is 2,432 + 4,160 + 130 = 6,722.

    - PROTEINS supplied three input channels whereas NCI1 supplies 37. The additional 34 input channels therefore contribute 34 × 64 = 2,176 additional first-layer weights, exactly accounting for the difference between the 4,546-parameter PROTEINS GCN and the 6,722-parameter NCI1 GCN.

- The measured training-pass time was 659.9765 seconds, with a mean of 0.6600 seconds per epoch.

    - The same established runtime convention applies, covering the training-pass operations while excluding validation, development-test evaluation, selected-state copying and restoration, progress printing, model-state saving and result writing.

    - NCI1 is therefore materially more expensive under the same development procedure than the preceding PROTEINS run. This observed runtime is retained for Stage 6 experiment-budget planning rather than used to change the scientific protocol.

- The restored selected state was saved as results/development_gcn_nci1_seed0.pt and the associated settings, complete development partitions, selection evidence, development-test metrics, parameter counts, runtime, device and source provenance were saved as results/development_gcn_nci1_seed0.json.

- Commit: 6.2 inspected NCI1 and recorded its development fit





# 6.3 Run Budget and Final Scope

- Fixed the investigation at four connected research questions evaluated on MUTAG, PROTEINS and NCI1.

    - RQ1 asks how the number of first-layer GAT attention heads affects graph-classification performance when the total first-layer representation width remains fixed at 64.

    - The compared head counts are 1, 2, 4 and 8. Their channels per head are therefore 64, 32, 16 and 8 respectively.

    - RQ2 compares the standard eight-head reference GAT with standard general-form GATv2 under the same graph-classification pipeline.

    - RQ2 is a comparison between the two complete attention formulations rather than a parameter-matched causal isolation of dynamic ranking, because the standard GAT and GATv2 implementations do not have identical parameterisations.

    - RQ3 compares the learned reference GAT with a fresh GAT trained while its neighbourhood attention coefficients are constrained to be uniform.

    - RQ4 examines the fitted reference GAT itself, first measuring how far its learned neighbourhood coefficients depart from uniform weighting and then measuring the sensitivity of its held-out predictions to replacing those learned coefficients with uniform weighting without retraining.

    - The four questions therefore progress through head organisation, scoring formulation, whether unequal weighting is useful during fitting, and what unequal weighting is actually learned by the fitted reference model.

- Fixed MUTAG, PROTEINS and NCI1 as the complete assessment dataset set.

    - Every research question uses the same three datasets.

    - Using the same dataset set throughout keeps the different attention investigations directly comparable and avoids introducing a dataset change as an additional factor between research questions.

    - No additional dataset is included in the assessment matrix.

- Retained all nine planned model configurations.

    - GCN, GraphSAGE and GIN remain contextual message-passing baselines.

    - GAT with 1, 2, 4 and 8 first-layer heads provides the RQ1 head-count comparison.

    - The eight-head GAT is the reference GAT used elsewhere in the investigation.

    - GATv2 provides the second principal learned-attention formulation.

    - Uniform GAT retains the GAT transformations and classifier while constraining the attention-scoring parameters so that neighbourhood softmax coefficients are uniform.

    - GraphSAGE and GIN were retained rather than removing contextual models solely to reduce runtime, preserving a broader reference picture for the principal attention models.

- Reconsidered the original 1,000-epoch assessment allowance before any cross-validation assessment results were produced.

    - The decision was treated as a prospective protocol decision rather than reducing individual models after observing favourable or unfavourable assessment outcomes.

    - A common epoch budget was retained across all configurations so training allowance itself does not become a model-specific experimental factor.

    - A systematic development-only convergence audit was used to compare shorter common budgets against the best validation state available under the previous 1,000-epoch allowance.

- experiments/inspections/inspect_epoch_budget.py evaluated all nine planned configurations on all three assessment datasets.

    - The resulting matrix contained 9 × 3 = 27 complete development trajectories.

    - Every trajectory still ran for the complete 1,000 epochs. The shorter cutoffs therefore changed only which earlier validation states were compared, not how far the audit trajectories themselves trained.

    - Validation was evaluated after every epoch using the existing graph-mean cross-entropy selection criterion.

    - For each trajectory, the audit retained the best validation-selected state available by epochs 100, 250, 500 and 1,000.

    - A named validation_sort_key function ordered recorded states by validation loss and then epoch so the earliest epoch was retained when two records had exactly equal loss.

    - The audit did not need to clone or restore model states because its purpose was to compare validation trajectories and candidate epoch budgets rather than produce assessment predictions.

- The epoch-budget audit used only development training and validation evidence.

    - Development-test results were not used to choose the epoch budget.

    - No cross-validation outer-test result existed when the 500-epoch decision was made.

    - The budget was fixed before CV outcomes, but development validation used the same benchmark datasets later partitioned for CV.

    - The report therefore describes the budget as informed by development validation on reused benchmark data. Fresh CV fits and disjoint partitions within each fold do not make the earlier global budget decision independent of those data.

    - This information use is recorded in the methodology and its assessment-independence limitation, without changing the accepted 500-epoch budget.

- An earlier audit run experienced machine interruptions.

    - That run reported a mean 500-epoch validation-loss gap of approximately 0.00447 and a maximum gap of approximately 0.05185.

    - Its interruption-affected elapsed time was unsuitable for runtime planning.

    - The clean rerun below provides the values used for the recorded budget analysis and elapsed-time projection; the two runs are kept distinct.

- The clean audit rerun completed all 27 trajectories successfully on CUDA.

    - Total elapsed time was 7,928.40 seconds, approximately 2.20 hours.

    - The result was saved as results/development_epoch_budget_audit.json.

    - The run reported source commit 834ca47202de75bd64cd89dd4bec17f36d2ccb70.

    - The reported source hash identifies repository HEAD at execution time. It is not by itself evidence that the working tree contained no uncommitted changes, so source provenance is interpreted together with the repository state rather than overstated.

- The 250-epoch cutoff was materially weaker than the chosen 500-epoch allowance.

    - By epoch 250, 11 of the 27 trajectories had already reached exactly the same minimum validation loss later available by epoch 1,000.

    - The median gap between the best loss available by epoch 250 and the eventual 1,000-epoch minimum was approximately 0.00355.

    - The mean 250-epoch gap was approximately 0.00827.

    - The largest observed 250-epoch gap was approximately 0.06563.

    - These results showed that 250 epochs would provide a substantially cheaper allowance, but the shorter budget more frequently excluded later lower-validation-loss states.

- The 500-epoch cutoff retained more of the validation-loss behaviour observed under the original allowance.

    - By epoch 500, 16 of the 27 trajectories had already reached exactly the same minimum validation loss later available by epoch 1,000.

    - The median 500-epoch validation-loss gap was exactly zero.

    - The mean gap was approximately 0.00476.

    - The largest observed gap was approximately 0.06177.

    - The zero median means that at least half of the audited trajectories had no further improvement in their selected validation loss after epoch 500.

    - It does not mean every trajectory had converged by that point.

- MUTAG GIN produced the largest observed difference between the 500-epoch and 1,000-epoch selections.

    - Its best validation state available by epoch 500 occurred at epoch 465 with loss 0.212927.

    - Its overall 1,000-epoch minimum occurred at epoch 682 with loss 0.151158.

    - The resulting gap of approximately 0.06177 demonstrates why the 500-epoch allowance is described as a computationally justified common budget rather than a guarantee that every configuration reaches its eventual minimum within 500 epochs.

- Later decreases in validation cross-entropy did not necessarily correspond to higher validation accuracy.

    - PROTEINS GIN, for example, had validation accuracy 0.8108 at the best state available by epoch 500, whereas the lower-loss state selected at epoch 540 had validation accuracy 0.7658.

    - This is not a conflict in the selection procedure because checkpoint selection is defined by graph-mean cross-entropy rather than validation accuracy.

    - The observation reinforces the need to distinguish the declared optimisation-selection quantity from other reported metrics rather than retrospectively preferring whichever epoch has the highest accuracy.

- Fixed 500 epochs as the common assessment training allowance.

    - Every assessment fit trains for all 500 epochs.

    - There is no model-specific epoch budget.

    - There is no patience-based early stopping.

    - Validation is evaluated after every epoch.

    - The selected state is the earliest state achieving the strict minimum graph-mean validation cross-entropy within the 500-epoch allowance.

    - Training continues after the selected epoch until all 500 epochs are complete, after which the selected state is restored for outer-test assessment.

    - The 500-epoch allowance therefore separates a fixed computational budget from validation-based state selection.

- Fixed assessment at stratified five-fold outer cross-validation.

    - Fold IDs are 0 through 4.

    - Each graph appears in the outer-test role exactly once across the five folds.

    - One outer fold is approximately 20 percent of the complete dataset.

    - Approximately one eighth of the remaining approximately 80 percent is reserved for validation.

    - This produces an intended whole-dataset allocation of approximately 70 percent fitting, 10 percent validation and 20 percent outer test in each fold.

    - The exact integer allocation was left to Stage 6.4 so that it could be implemented and checked directly rather than describing approximate percentages as exact graph counts.

- Fixed a deterministic seed schedule with separate roles for partitioning and model fitting.

    - Outer-fold construction uses split seed 0.

    - Fold-local validation construction uses seed 1000 + fold_id.

    - Model initialisation and fitting-loader randomness use seed 2000 + fold_id.

    - Corresponding configurations therefore use exactly the same fold identities and partition construction while retaining one predetermined training realisation for each fold.

    - No seed search or repeated-initialisation grid is introduced.

- Fixed the assessment summary to the five held-out fold results.

    - The individual fold values are retained rather than reporting only an aggregate.

    - The arithmetic mean summarises the five outer-test values.

    - Sample standard deviation is calculated with ddof=1, corresponding to a denominator of 4 for five folds.

    - The folds are treated as the five cross-validation assessments rather than incorrectly described as independent datasets, because their fitting partitions overlap.

- Fixed the complete optimisation matrix at 135 fits.

    - GCN, GraphSAGE, GIN, reference GAT and GATv2 across three datasets and five folds give 5 × 3 × 5 = 75 reference fits.

    - The additional one-head, two-head and four-head GAT configurations required for RQ1 contribute 3 × 3 × 5 = 45 fits.

    - The eight-head GAT is reused from the reference matrix rather than fitted again for RQ1.

    - RQ2 requires no additional optimisation because both GAT and GATv2 are already present in the reference matrix.

    - Uniform GAT contributes 1 × 3 × 5 = 15 additional fits for RQ3.

    - RQ4 reuses the 15 selected reference-GAT states and therefore adds no optimisation fits.

    - The total is 75 + 45 + 15 = 135 fits.

- The 500-epoch and five-fold decisions reduce the nominal optimisation workload substantially compared with the earlier design.

    - The earlier ten-fold, 1,000-epoch matrix implied 270 fits and 270,000 nominal fit-epochs.

    - The locked matrix contains 135 fits × 500 epochs = 67,500 nominal fit-epochs.

    - 67,500 is exactly one quarter of 270,000.

    - This reduction preserves all nine scientific configurations while reducing repeated outer folds and the common maximum training allowance rather than deleting model families from the investigation.

- Used the clean convergence audit to form a practical runtime estimate.

    - Twenty-seven 1,000-epoch development trajectories required approximately 2.20 hours.

    - The assessment contains five times as many fits but half as many epochs per fit.

    - Direct scaling therefore gives approximately 2.20 × 5 × 0.5 = 5.5 hours as an approximate elapsed-time estimate under similar machine conditions.

    - The estimate is planning evidence rather than a promised duration because assessment partition sizes, validation, result recording, selected-state handling and machine conditions differ from the development audit.

    - The audit total measures elapsed time and is distinct from training_seconds in ordinary fit records, which measures only the timed training passes. The 5.5-hour projection is therefore not a training-pass-only measurement.

    - RQ4 inference and analysis add time beyond this projection of the optimisation matrix.

- Fixed the first component of RQ4 as a graph-weighted measurement of learned attention non-uniformity.

    - RQ4 uses the validation-selected eight-head reference GAT from every dataset and outer fold, giving 15 fitted reference states.

    - Only the corresponding outer-test graphs are analysed for each fitted state.

    - Both GAT layers are examined separately.

    - For receiver i and head h with m_i greater than one incoming entries, normalised entropy is H_i,h = -sum_j alpha_i,j,h log(alpha_i,j,h) / log(m_i).

    - Departure from uniformity is D_i,h = 1 - H_i,h.

    - Perfectly uniform weighting gives D_i,h = 0.

    - Receivers with one incoming entry are excluded from this entropy calculation because log(1) is zero and no weighting choice exists. Their counts are still retained.

    - In the first GAT layer, the per-head departure values are averaged to obtain the primary receiver-level value.

    - Eligible receivers are then averaged within each graph.

    - Graphs are averaged equally within a fold so large graphs do not dominate the dataset-level value simply because they contain more receiver nodes.

    - The resulting five fold-level values are retained and summarised using their arithmetic mean and sample standard deviation.

- Fixed the second component of RQ4 as an intervention on the same fitted reference-GAT states.

    - The same selected model state and same outer-test graphs are evaluated before and after intervention.

    - After loading the selected state, conv1.att_src, conv1.att_dst, conv2.att_src and conv2.att_dst are set to zero.

    - No optimiser is created and no retraining occurs after the intervention.

    - Message transformations, biases, classifier parameters, topology, self-loop handling and head organisation remain unchanged.

    - With equal attention logits and zero attention dropout, neighbourhood softmax assigns equal weight to the effective incoming entries.

    - The intervention therefore removes the learned unequal attention scoring while preserving the remainder of the fitted model.

- Fixed three predictive sensitivity endpoints for the RQ4 intervention.

    - Cross-entropy change is defined as intervention loss minus learned-attention loss.

    - Accuracy change is defined as intervention accuracy minus learned-attention accuracy.

    - Prediction-flip rate is the fraction of outer-test graphs whose predicted class changes after the intervention.

    - Original graph identities, true labels, learned-attention predictions and intervention predictions must remain aligned so that graph-level changes can be inspected directly.

    - The intervention is interpreted as fitted-model sensitivity to removing learned unequal weighting.

    - It is not treated as evidence that attention coefficients are faithful explanations or that individual neighbours have causal importance in the underlying scientific domain.

- Kept the assessment methodology focused on controlled comparison rather than separate benchmark optimisation for every model.

    - Hidden width, optimiser, learning rate, weight decay, batch size and other established shared settings remain fixed.

    - No assessment learning-rate grid, width grid, dropout grid, epoch grid or seed search is introduced.

    - Conclusions are therefore restricted to the behaviour of the compared mechanisms under the declared matched pipeline rather than claiming globally optimal performance for every model family.

- The research questions, datasets, configurations, epoch allowance, fold count, seed roles and RQ4 analysis were fixed before cross-validation assessment results were produced.

    - Stage 6.4 therefore became an implementation and verification stage rather than another opportunity to change the scientific scope after observing held-out results.

- Commit: 6.3 fixed the experiment budget and final scope





# 6.4 Implement and Lock Final Cross-Validation

- Preserved the established development data utilities before changing the active data path for cross-validation.

    - The previous src/data.py was copied to archive/src/data.py.

    - The archive mirrors the original repository path rather than introducing a separate archive hierarchy for one stage.

    - The archived file is retained as historical source and is not imported by the active experiment code.

    - The active src/data.py continues to contain the earlier development split logic alongside the new cross-validation functions, so the historical behaviour remains understandable while the current assessment path is explicit.

- Added stratified_folds to construct the outer cross-validation folds.

    - The function first creates one list of original graph indices for each processed class.

    - enumerate(dataset) supplies the stable original dataset index of every graph, while graph.y.item() identifies the processed class used for stratification.

    - np.random.default_rng(seed) creates the local NumPy random generator used for outer-fold shuffling without relying on unrelated global random state.

    - Each class list is converted to a NumPy array and shuffled independently using the same generator.

    - The shuffled indices are distributed cyclically across the requested number of folds.

    - fold_id advances after every graph and is not reset when the next class begins.

    - Carrying the fold position across classes avoids unnecessarily favouring fold 0 whenever a class count is not divisible by five and helps balance complete fold sizes as well as class-specific counts.

    - Every completed fold is sorted before being returned so the stored partition representation follows original dataset-index order rather than shuffled allocation order.

- The outer split is parameterised by num_folds and seed rather than hard-coding five folds and seed 0 inside src/data.py.

    - train_cv.py declares num_folds=5 and split_seed=0 as experiment settings.

    - This keeps the scientific choice visible in the experiment runner while stratified_folds remains a simple reusable partition function.

    - Fold IDs are generated as 0 through 4 from the configured number of folds.

- Added stratified_validation_split to create the fitting and validation partitions inside one outer-fold remainder.

    - The function receives only the indices remaining after the outer-test fold has been removed.

    - It rebuilds class-specific index lists from that remainder so validation allocation remains stratified independently inside each outer fold.

    - This separation prevents outer-test graphs from becoming candidates for validation.

    - A new local NumPy generator is created from the supplied fold-specific validation seed.

- Fixed the exact classwise validation-size rule as int(n / 8 + 0.5).

    - n is the number of graphs from one class remaining after removal of the outer-test fold.

    - Dividing by 8 targets one eighth of the approximately 80 percent outer remainder.

    - One eighth of 80 percent is 10 percent of the complete dataset, producing the intended approximately 70/10/20 fitting, validation and outer-test allocation.

    - Adding 0.5 before int() implements ordinary nearest-integer rounding explicitly rather than depending on Python round() behaviour at exact half values.

    - The result is bounded between 1 and n - 1 so a represented class cannot lose every remaining graph to validation and cannot contribute zero validation graphs.

    - The current datasets contain far more than two outer-remainder examples per class, but an explicit ValueError rejects a class with fewer than two because a valid fit/validation division would then be impossible.

- Fitting and validation indices are sorted after the classwise allocation.

    - Sorting does not change which graphs belong to either partition.

    - It gives the saved partitions one stable original-index representation and makes graph identity easier to inspect later.

    - Training randomness is introduced subsequently by the fitting DataLoader rather than by preserving an arbitrary shuffled order in the saved fitting-index list.

- Added experiments/inspections/inspect_cv_splits.py so the new partition contract could be checked with one clean terminal command.

    - The inspection was run with python -m experiments.inspections.inspect_cv_splits.

    - class_counts iterated over supplied original dataset indices and counted the processed target class of every selected graph.

    - The inspection therefore reported both total partition sizes and class-specific counts rather than checking only that the arithmetic totals looked plausible.

    - set objects were used for the integrity checks because membership and overlap between graph-index collections are the relevant properties.

    - isdisjoint verified that fitting, validation and outer-test partitions contained no common graph index.

    - The pipe operator between sets formed their union so the inspection could verify that the three partitions collectively reproduced the complete set of dataset indices.

- Every inspected MUTAG fold passed the partition checks.

    - Folds 0, 1 and 2 each contained 131 fitting graphs, 19 validation graphs and 38 outer-test graphs.

    - Their fitting class counts were [44, 87], validation class counts were [6, 13] and test class counts were [13, 25].

    - Folds 3 and 4 each contained 132 fitting graphs, 19 validation graphs and 37 outer-test graphs.

    - Their fitting class counts were [45, 87], validation class counts were [6, 13] and test class counts were [12, 25].

    - Every fold reported Disjoint: True and Complete: True.

    - The final Outer test partition: True check established that the five test folds collectively contained all 188 MUTAG graphs exactly once.

- Every inspected PROTEINS fold also passed the partition checks.

    - Folds 0, 1 and 2 each contained 779 fitting graphs, 111 validation graphs and 223 outer-test graphs.

    - Their fitting class counts were [464, 315], validation class counts were [66, 45] and test class counts were [133, 90].

    - Folds 3 and 4 each contained 780 fitting graphs, 111 validation graphs and 222 outer-test graphs.

    - Their fitting class counts were [465, 315], validation class counts were [66, 45] and test class counts were [132, 90].

    - Every fold reported Disjoint: True and Complete: True.

    - Outer test partition: True established that the five test folds collectively contained all 1,113 PROTEINS graphs exactly once.

- Every inspected NCI1 fold passed the same checks.

    - Every fold contained exactly 2,877 fitting graphs, 411 validation graphs and 822 outer-test graphs.

    - Folds 0, 1 and 2 had fitting class counts [1437, 1440], validation class counts [205, 206] and test class counts [411, 411].

    - Folds 3 and 4 had fitting class counts [1438, 1439], validation class counts [205, 206] and test class counts [410, 412].

    - The slight class-count difference in folds 3 and 4 follows from distributing class totals that are not both divisible exactly by five.

    - Every fold reported Disjoint: True and Complete: True.

    - Outer test partition: True established that all 4,110 NCI1 graphs occur exactly once in the outer-test role.

- The realised fold sizes confirmed the intended approximate whole-dataset allocation without requiring equal integer counts in every fold.

    - MUTAG uses 131 or 132 fitting graphs, 19 validation graphs and 37 or 38 test graphs.

    - PROTEINS uses 779 or 780 fitting graphs, 111 validation graphs and 222 or 223 test graphs.

    - NCI1 uses 2,877 fitting graphs, 411 validation graphs and 822 test graphs in every fold.

    - These exact sizes are consequences of class-stratified integer allocation rather than percentages being rounded only after the complete dataset is split.

- Reorganised inspection utilities under experiments/inspections so diagnostic scripts remain separate from experiment runners.

    - Dataset inspectors now live under experiments/inspections/datasets.

    - Model inspectors now live under experiments/inspections/models.

    - General methodological inspections such as inspect_cv_splits.py and inspect_epoch_budget.py live directly under experiments/inspections.

    - The active experiments directory is therefore reserved for experiment-level runners and orchestration rather than accumulating unrelated inspection scripts.

    - The reorganised CV inspection was rerun successfully through its new module path and reproduced the same partition output.

- Stage 6.4 initially extended src/evaluation.py with evaluate_with_predictions alongside the original two-value evaluate. The pre-run correction consolidated the active file around one evaluate function.

    - The original two-value implementation had already been preserved in archive/src/evaluation.py before the prediction-preserving addition. That archived original remains unchanged.

    - The active evaluate now performs the same calculation and returns the same four values as the former evaluate_with_predictions, removing the duplicate metric loop.

    - train_model and inspect_epoch_budget.py unpack all four values and use the loss and accuracy; the CV runner also uses the returned prediction and label lists.

    - model.eval() switches the model to evaluation mode before held-out inference.

    - torch.no_grad() prevents gradient construction because evaluation does not perform optimisation.

    - Each graph batch is moved to the selected device before the model receives x, edge_index and batch.

    - F.cross_entropy calculates the same per-batch classification loss used by the existing evaluation path.

    - Multiplying loss.item() by graph_batch.num_graphs before accumulation ensures the final division by total_graphs gives graph-mean cross-entropy rather than an unweighted average of differently sized minibatch means.

    - logits.argmax(dim=1) converts class logits into one predicted class ID per graph.

    - Accuracy is accumulated as the number of correct graph predictions divided by the total number of graphs.

- The active evaluate preserves graph-level prediction evidence as well as the two metrics.

    - predictions.cpu().tolist() moves predicted class IDs to CPU and converts them into ordinary Python values suitable for JSON recording.

    - graph_batch.y.cpu().tolist() does the same for the true labels.

    - Predictions and labels are appended in loader order.

    - Outer-test DataLoaders use shuffle=False, while test_indices are stored in sorted original-index order.

    - The saved test_indices, predictions and labels are therefore aligned position by position.

    - This graph-level alignment is required later for the RQ4 intervention, where predictions from the fitted reference GAT and its uniform-attention intervention must be compared on exactly the same held-out graphs.

- Extended src/recording.py so result filenames can include explicit fold identity.

    - get_result_path already constructs paths from result type, model, dataset, optional variant and training seed.

    - When fold_id is present in the supplied settings, the function now inserts _fold followed by the fold number before the seed component.

    - A standard reference GAT fold can therefore use a path such as results/cross_validation_gat_mutag_fold0_seed2000.json.

    - A one-head variant can use results/cross_validation_gat_mutag_heads1_fold0_seed2000.json.

    - Historical development settings do not contain fold_id, so their established filenames remain unchanged.

- The existing model-state recording path remains separate from the JSON evidence record.

    - save_model_state stores model.state_dict(), containing the learned model tensors rather than serialising the complete Python model object.

    - The state can later be restored into the same architecture with load_state_dict.

    - This is important for RQ4 because the exact validation-selected reference-GAT state from each fold must later be loaded for attention analysis and intervention without retraining.

    - The binary file is opened with mode xb, so an existing state file is not silently overwritten.

- Added the cross-validation experiment runner as experiments/run_cv.py and renamed it to experiments/train_cv.py during the pre-run corrections.

    - train_cv.py is a descriptive counterpart to the retired single-split train.py and identifies the cross-validation training procedure.

    - The active run_cv.py path was removed by the rename, and ablation.py now imports experiments.train_cv.

    - The runner directly owns the current assessment settings rather than importing them from the historical single-split development runner.

    - Shared settings are hidden_dim 64, learning rate 0.01, weight decay 0.0005, 500 epochs, batch size 32, split seed 0 and five folds.

    - dataset and training seed are not stored as meaningless common defaults because they are assigned by the dataset and fold loops respectively.

    - model_settings contains only configuration-specific additions for the five reference models.

    - GCN, GraphSAGE and GIN require no additional settings beyond their model identity.

    - Reference GAT uses 8 heads.

    - GATv2 uses 8 heads with share_weights=False.

- train_cv.py declares MUTAG, PROTEINS and NCI1 as the three assessment datasets and GCN, GraphSAGE, GIN, GAT and GATv2 as the five reference models.

    - The Cartesian combination of those lists produces the 15 dataset/model combinations required for Stage 6.5.

    - Each combination is then evaluated across fold IDs 0 through 4, producing the planned 75 reference fits.

    - The runner therefore expresses the complete reference matrix directly without manually listing 75 separate jobs.

- run_cross_validation receives one complete model/dataset settings dictionary and the fold IDs to execute.

    - load_dataset loads the selected dataset once for that model/dataset combination.

    - stratified_folds reconstructs the deterministic outer folds from num_folds and split_seed.

    - fold-specific settings are copied from the supplied settings so one fold cannot mutate the settings used by another.

    - fold_id records explicit outer-fold identity.

    - validation_split_seed is assigned as 1000 + fold_id.

    - training seed is assigned as 2000 + fold_id.

- The runner derives every fold's output paths before beginning training.

    - get_result_path constructs the fold-specific JSON path.

    - Replacing the .json suffix with .pt gives the paired model-state path.

    - prepare_result_path checks both paths before the fold is added to the list of executable runs.

    - All requested fold paths are therefore checked before optimisation begins.

    - If one requested result or state already exists, execution fails before spending time fitting earlier folds and then discovering the conflict later.

    - This preserves the established project rule that recorded evidence is never silently overwritten.

- The source commit and compute device are recorded once before executing the prepared folds.

    - get_source_commit calls git rev-parse HEAD so every saved result identifies the repository commit visible to Git at execution time.

    - torch.cuda.is_available() selects CUDA when available and otherwise falls back to CPU.

    - The selected device type is printed before fitting begins.

    - CUDA device name is also retained in the saved result when CUDA is used.

- Retained scientific qualifiers while simplifying temporary implementation names.

    - outer_folds names the assessment-fold collection in src/data.py, train_cv.py and inspect_cv_splits.py. The runner and split inspector print Outer fold.

    - An outer assessment fold remains an outer fold when its inner selection uses a single validation holdout rather than inner cross-validation.

    - Names such as outer_split_seed, validation_split_seed, selected_epoch and reuse_reference retain their distinct meanings. Existing fold_id, fold_ids, num_folds and stratified_folds remain clear in context.

    - The cleanup removes unnecessary project-timeline labels such as final_settings without removing scientific distinctions.

- Each outer fold is reconstructed explicitly from original dataset indices.

    - outer_folds[fold_id] supplies the current test indices.

    - A set of those indices supports efficient membership checking while constructing the outer remainder.

    - remainder_indices contains every dataset index that is not part of the current outer-test fold.

    - stratified_validation_split then divides only that remainder into fitting and validation indices using the fold-specific validation seed.

- Model-training randomness is fixed separately from partition randomness.

    - set_seed(2000 + fold_id) seeds Python random, NumPy and PyTorch before model construction.

    - A separate torch.Generator is created and manually seeded with the same training seed for the fitting DataLoader.

    - The fitting loader uses shuffle=True, so training minibatch order changes deterministically according to the assigned fold seed.

    - Validation and outer-test loaders use shuffle=False because their purpose is deterministic assessment rather than stochastic optimisation.

    - Corresponding configurations use the same training seed for a given fold, preserving one predetermined training realisation rather than searching for favourable seeds.

- Model construction and optimisation reuse the existing shared lower-level implementation.

    - build_model receives the fold settings, dataset input width and number of classes.

    - The selected model is moved to the chosen device before optimisation.

    - Adam receives only parameters whose requires_grad flag is True.

    - This matters for Uniform GAT because its attention-scoring parameters are deliberately frozen while the remaining trainable transformations and classifier are still optimised.

    - Learning rate and weight decay come directly from the shared assessment settings.

- Every valid fold uses the shared train_model selection path for all 500 epochs.

    - train_model performs one training pass per epoch and evaluates validation after every epoch.

    - The selected state changes only when validation loss is strictly lower than the previous minimum.

    - An exact tie therefore leaves the earlier selected state unchanged.

    - The selected state tensors are preserved during training.

    - Optimisation still continues for the complete 500 epochs.

    - At the end of training, the selected state is restored into the model before run_cross_validation receives control again.

    - The subsequent outer-test evaluation therefore uses the validation-selected state rather than the epoch-500 state automatically.

- Corrected numerical-failure handling in train_model before CV execution.

    - math.isfinite checks the returned training and validation losses before checkpoint selection. It returns False for NaN and positive or negative infinity.

    - A non-finite loss now raises a ValueError identifying whether training or validation failed and at which epoch.

    - Previously, a NaN validation loss could fail the strict improvement comparison while an earlier finite checkpoint remained available. Training could then restore that earlier state and record selected metrics without preserving the invalid intervening epochs in the JSON.

    - The guard stops such an invalid run for investigation. Valid runs still complete all 500 epochs, use strict validation-loss improvement and retain the earliest exact tie.

    - No automatic retry, shorter successful run or omission from the five-fold result group is introduced.

- Outer-test assessment is performed only after the validation-selected state has been restored.

    - evaluate returns graph-mean cross-entropy, accuracy, graph-level predicted classes and graph-level true labels, in that order.

    - The runner checks test_loss with math.isfinite and raises a ValueError identifying the outer fold if the loss is NaN or infinite. This occurs before either the state or JSON is saved.

    - No outer-test information is supplied to train_model or used for state selection.

    - The outer-test partition therefore remains an assessment partition rather than becoming part of optimisation or checkpoint choice.

- Each fold records total and trainable parameter counts.

    - total_parameters sums numel() over every model parameter.

    - trainable_parameters sums only parameters with requires_grad=True.

    - For ordinary models these values are expected to be equal.

    - For Uniform GAT the distinction records the frozen attention-scoring parameters explicitly rather than treating the constrained model as if every stored parameter were trainable.

- The runtime calculation preserves the established training-pass convention.

    - training_seconds is returned by train_model.

    - mean_seconds_per_epoch divides that value by the fixed 500 completed epochs.

    - The saved convention states that the timing covers training-loader iteration, device transfer, forward pass, loss, backward pass, optimiser update and training metric calculation and accumulation.

    - Validation, outer-test assessment, selected-state copying and restoration, progress printing, model-state saving and result writing remain outside that timing quantity.

- Each cross-validation JSON records the information required to reproduce and audit one fold.

    - settings contains the model configuration together with dataset, fold ID, validation split seed and training seed.

    - dataset records the settled TUDataset loading policy.

    - feature_policy states explicitly that categorical node labels/features and connectivity are used while continuous node attributes and edge features are excluded.

    - partitions records the outer split seed, validation split seed and complete original-index lists for fitting, validation and outer test.

    - selection records the declared criterion, tie rule, absence of early stopping, selected epoch, completed epoch count, selected validation loss and selected validation accuracy.

    - outer_test records the held-out loss, accuracy, predictions and labels.

    - parameters, runtime, device, source_commit and model_state_path preserve the remaining execution provenance.

- Each fold saves the selected model state before writing its JSON record.

    - The .pt file contains the state restored by train_model, so it corresponds to the validation-selected epoch rather than simply the last optimisation epoch.

    - The paired JSON points to that state path.

    - The reference GAT .pt files are therefore directly reusable for the later RQ4 fitted-attention analysis without refitting the model.

- The runner prints the most important evidence after every fold.

    - Fold identity, fitting size, validation size, test size, training seed and validation split seed are printed before training.

    - Selected epoch, validation loss and validation accuracy are printed after fitting.

    - Outer-test loss and accuracy are printed separately so selection evidence is not confused with held-out assessment evidence.

    - Parameter counts, training time, mean seconds per epoch and both saved paths are also printed.

- Updated experiments/ablation.py so later attention variants reuse run_cross_validation instead of maintaining their own training framework.

    - The variant list contains GAT heads 1, 2, 4 and the eight-head reference configuration, the reference GATv2 configuration and Uniform GAT.

    - get_variant_settings begins from the shared train_cv settings, applies the standard settings for the selected model and then applies only the settings specific to that variant.

    - This ordering gives variant-specific settings the final say without duplicating the complete common configuration in every variant dictionary.

- The head-count configurations preserve total first-layer representation width at 64.

    - One head uses 64 channels per head.

    - Two heads use 32 channels per head.

    - Four heads use 16 channels per head.

    - Eight heads use 8 channels per head.

    - Multiplying channels per head by head count gives 64 in every case.

    - The ablation inspection printed these derived values directly so the width-control condition was checked before any RQ1 fitting begins.

- The ablation runner distinguishes configurations requiring new optimisation from configurations already present in the reference matrix.

    - GAT heads 1, 2 and 4 are marked as additional cross-validation fits.

    - The eight-head reference GAT is marked as reusing its existing cross-validation results.

    - GATv2 is also marked as reusing its existing cross-validation results.

    - Uniform GAT is marked as an additional cross-validation fit.

    - run_variant rejects a configuration marked for reference reuse rather than accidentally retraining it under a second path.

- The ablation inspection also checked fold-aware result naming without starting training.

    - The command python -m experiments.ablation printed the effective settings for each configuration.

    - All configurations reported 500 epochs, batch size 32, split seed 0 and five folds.

    - Example paths used MUTAG fold 0 and training seed 2000.

    - The one-head example path was results/cross_validation_gat_mutag_heads1_fold0_seed2000.json.

    - The two-head example path was results/cross_validation_gat_mutag_heads2_fold0_seed2000.json.

    - The four-head example path was results/cross_validation_gat_mutag_heads4_fold0_seed2000.json.

    - The reference-GAT example path was results/cross_validation_gat_mutag_fold0_seed2000.json.

    - The GATv2 example path was results/cross_validation_gatv2_mutag_fold0_seed2000.json.

    - The Uniform GAT example path was results/cross_validation_gat_mutag_uniform_fold0_seed2000.json.

- Removed the active cross-validation code's dependency on the historical development runner.

    - An initial version of run_cv.py reused settings from experiments/train.py, which caused historical development defaults to leak into current inspection output even though the actual dataset loop would later overwrite them.

    - The current train_cv.py instead owns the active assessment settings and model settings directly.

    - ablation.py imports those current definitions from train_cv.py.

    - This gives the active assessment code one clear source for the current protocol and avoids depending on a runner whose purpose was the earlier single-split development phase.

- Confirmed that no active Python file still imports experiments.train.

    - Get-ChildItem experiments -Recurse -Filter *.py | Select-String "experiments.train" produced no matches after the dependency was removed.

    - experiments/train.py was then moved to archive/experiments/train.py.

    - The earlier ablation implementation had already been retained under archive/experiments/ablation.py.

    - The archive therefore preserves the superseded development implementations while the active experiments package contains the current cross-validation path.

    - archive/src/data.py, archive/src/evaluation.py and archive/src/recording.py preserve the corresponding original implementations from before the CV additions.

    - The archive mirrors the source directories directly. Active code does not import it, and the archived files are historical source snapshots rather than a separately maintained runnable package.

- Verified the active modules after archiving the development runner, before the later rename to train_cv.py.

    - python -c "import experiments.run_cv; import experiments.ablation; print('imports passed')" printed imports passed.

    - python -m experiments.ablation then executed successfully from the reorganised active code.

    - The inspection did not start optimisation and confirmed that the active configuration no longer inherits the retired development runner.

- Updated README.md to describe the implemented Stage 6.4 code, the train_cv.py entry point, inspection locations, archive and outstanding reference fits.

    - The external roadmap and system prompt now agree on the active configuration owner, four-value evaluator, numerical-failure guards, clean audit values, interrupted-run history and elapsed-time wording.

    - The governing log convention now explicitly retains the Commit: prefix requested in the handoff. A suggested summary is not evidence that Git has been run.

- Checked the pre-run corrections without executing cross-validation fits.

    - All 33 Python files parsed successfully, and the active imports and evaluator callers matched the renamed runner and four-value interface.

    - A source comparison confirmed that the consolidated evaluate retained the complete calculation of evaluate_with_predictions and that the data-partition logic changed only its fold variable names and requested ValueError formatting.

    - Controlled training checks used stand-ins for numerical operations. A finite four-epoch trajectory with equal minimum losses at epochs 2 and 3 completed all four epochs and restored the epoch-2 parameter and persistent buffer.

    - NaN, positive infinity and negative infinity in either training or validation loss each raised the expected epoch-specific error. Equivalent outer-test checks rejected all three before either output was saved.

    - These checks exercised control flow, not neural-network computation. PyTorch and PyG were unavailable in the audit environment, so no new model fit or real-library execution is claimed.

    - The existing model definitions, archive and saved results were unchanged. No source commit was made in the extracted snapshot.

- Stage 6.4 therefore fixed both the data-partition contract and the execution contract before any reference cross-validation results were generated.

    - Outer folds, validation construction, seed roles, epoch allowance, checkpoint selection, test evaluation, graph-level prediction recording, state saving and result naming now follow one shared implementation.

    - Stage 6.5 can execute the 75 reference fits without making further methodological choices about how the cross-validation procedure works.

- Commit: 6.4 implemented and locked the cross-validation protocol





# 6.5 Final Reference Runs

- Executed the reference cross-validation matrix through python -m experiments.train_cv.

    - The reference models were GCN, GraphSAGE, GIN, GAT and GATv2.

    - Each model was evaluated on MUTAG, PROTEINS and NCI1 using the five outer folds established in Stage 6.4.

    - This produced 5 × 3 × 5 = 75 separate fits, organised into 15 dataset/model groups.

    - One group therefore means one model evaluated on one dataset across all five outer folds. For example, the GCN/MUTAG group contains five separately fitted GCN models.

- Corrected an evaluator interface mismatch encountered at the first validation call of the initial execution.

    - src/training.py expected evaluate to return validation loss, validation accuracy, predictions and labels.

    - The active src/evaluation.py still contained the earlier evaluate implementation returning only loss and accuracy, while the four-value implementation had been placed in the archive.

    - Python therefore raised ValueError: not enough values to unpack (expected 4, got 2).

    - The failure occurred during validation in the first GCN/MUTAG outer fold, before that fold could complete and save its result.

    - Restored the four-value evaluate implementation to active src/evaluation.py and retained the original two-value implementation in archive/src/evaluation.py.

    - The unused prediction and label returns in the validation call were assigned to underscores. Validation still used only loss and accuracy, while the shared evaluator also supplied predictions and labels when the runner assessed the outer-test partition.

    - This corrected the active function contract without changing the loss calculation, checkpoint-selection rule or experimental settings.

- Restored the full per-fold Settings display in experiments/train_cv.py.

    - The runner already printed the dataset and model at the start of each dataset/model group.

    - The added loop over fold_settings.items() printed the effective setting names and values before each fit, including model identity, dataset, shared hyperparameters and fold-specific seeds.

    - This made the actual configuration visible alongside the current outer fold instead of requiring the reader to refer back to an earlier group heading.

    - The change affected terminal reporting rather than model construction, optimisation or recorded scientific settings.

- The successful reference run used the fixed Stage 6.4 training and assessment procedure.

    - Every fit constructed a fresh model and optimiser and completed the common 500-epoch allowance.

    - Validation after every epoch selected the earliest state attaining the minimum validation cross-entropy.

    - The selected state was restored after all 500 epochs and evaluated once on the corresponding outer-test graphs.

    - The five fits within a group therefore assessed five different selected models on their respective held-out partitions. They were not repeated evaluations of one model or an ensemble.

    - Each graph appeared in one outer-test fold within each model/dataset group. Corresponding models used the same partition indices, allowing later comparisons on matched held-out graphs.

- The terminal output reached the final scheduled fit, GATv2 on NCI1 with outer-fold ID 4, and returned to PowerShell without a reported error.

    - This fit used 2,877 fitting graphs, 411 validation graphs and 822 outer-test graphs.

    - Its validation split seed was 1004 and its training seed was 2004, following the established 1000 + fold_id and 2000 + fold_id schedules.

    - Training reached epoch 500, but the selected state came from epoch 484.

    - The selected validation loss was 0.5622 and the selected validation accuracy was 0.7129.

    - The epoch-500 validation loss was 0.5706. Restoring epoch 484 therefore retained the lower-validation-loss state rather than automatically using the final optimisation state.

    - Epoch 484 was absent from the ten-epoch progress lines because checkpoint selection occurred every epoch while progress was printed every ten epochs.

- The restored GATv2 state for NCI1 outer fold 4 obtained outer-test loss 0.5498 and accuracy 0.7251, approximately 72.51%.

    - This accuracy described that fold's 822 held-out graphs. It was one of the five values contributing to the later GATv2/NCI1 summary.

    - The model contained 13,570 parameters, all trainable.

    - Its recorded training-pass time was 285.5012 seconds, giving 285.5012 / 500 = approximately 0.5710 seconds per training epoch.

    - The timing covered all 500 training passes, including those after the selected epoch. It did not measure only the time required to reach epoch 484 or the complete elapsed time of the program.

- The final fit saved results/cross_validation_gatv2_nci1_fold4_seed2004.pt and results/cross_validation_gatv2_nci1_fold4_seed2004.json.

    - The .pt file preserved the restored validation-selected model state.

    - The JSON preserved the associated effective settings, partition indices, selection evidence, outer-test predictions and labels, parameter counts, timing and execution provenance.

    - The paired filenames identify the model, dataset, outer fold and training seed belonging to the same fit.

- Confirmed that the completed reference run produced 75 JSON records and 75 paired model-state files.

    - These 150 files matched the expected output count for 75 reference fits.

    - The terminal output identified source commit c60bbd57fd1c7cc7e6bce3a4a51c6fdc54b0776f for the final model/dataset group.

    - File counts and normal termination established completion of the run. Checking every group's identities, settings, partitions and prediction-derived measurements remained the purpose of Stage 6.6.

    - The reference GAT and GATv2 records were retained for the later attention comparisons, and the selected reference-GAT states were retained for fitted-attention analysis without retraining.

- Commit: 6.5 recorded the reference cross-validation results





# 6.6 Necessary Summarisation

- Added experiments/summarise.py to turn the saved reference records into validated accuracy and resource summaries.

    - The script read the existing JSON records and loaded each dataset to check graph identities and labels.

    - It checked that every paired model-state file existed but did not load those weights, repeat model inference or start training.

    - One output row represented one dataset/model group containing five outer-fold fits.

    - The complete reference matrix therefore produced 15 rows in the accuracy table and 15 rows in the parameter and runtime table.

- Imported datasets, models, settings, model_settings and result_type from experiments/train_cv.py.

    - The summary used the same configuration owner as the reference runner rather than maintaining another independently editable copy of the scientific settings.

    - main copied the shared settings, applied the selected model's additional settings and assigned the current dataset and model.

    - load_group then added the expected outer-fold ID, validation split seed and training seed when checking each record.

    - Fold-specific settings were therefore allowed to differ according to the declared schedule while the common scientific recipe remained fixed.

- get_partitions reconstructed the expected partition dictionaries for each dataset.

    - stratified_folds reproduced the outer-test indices using the configured fold count and outer split seed.

    - For each outer fold, the function constructed the remainder and passed it to stratified_validation_split with seed 1000 + fold_id.

    - The returned dictionaries contained the expected fitting, validation and outer-test indices together with their partition seeds.

    - These dictionaries were constructed once per dataset and reused when checking every model on that dataset.

    - Comparing each saved partitions dictionary with its corresponding expected dictionary checked both graph membership and index order. This also established that compared models used the same partitions for a given outer fold.

- load_group selected the reference JSON files using ordinary directory and string operations.

    - os.listdir("results") returned the names inside the results directory, and sorted placed them in a consistent processing order.

    - startswith selected the current reference prefix, such as cross_validation_gcn_mutag_fold, while endswith(".json") selected JSON records.

    - This excluded historical development records, model-state files and separately named attention variants from the reference group.

    - os.path.join combined the directory and filename into a path suitable for the operating system.

    - os.path.normpath normalised paths before comparison, allowing Windows path separators to be compared consistently with the forward-slash paths stored in the records.

- The group reader used a dictionary whose keys were the recorded outer-fold IDs.

    - A second matching record claiming an already encountered fold ID raised an error rather than replacing the first record or being counted as another independent fold.

    - Comparing the dictionary's keys with the set of expected fold IDs required exactly folds 0 through 4.

    - The function returned the accepted records in numerical outer-fold order, which determined the order of the five accuracy columns.

    - Each record also had to match its expected purpose, complete effective settings and filename generated by get_result_path.

    - The recorded model_state_path had to identify the corresponding .pt file, and os.path.isfile checked that this file existed. This established the presence of the paired state rather than independently validating its contents.

- Checked that each record represented the intended dataset, feature policy and checkpoint-selection procedure.

    - The recorded dataset flags retained cleaned=False, use_node_attr=False and use_edge_attr=False.

    - The feature policy retained categorical node information and connectivity while excluding continuous node attributes and edge features.

    - Selection metadata had to specify minimum validation cross-entropy, the earliest exact tie rule, no early stopping and the configured 500 completed epochs.

    - The selected epoch had to fall within the completed training allowance.

    - Recorded training times and losses had to be finite and non-negative, and validation accuracy had to lie between zero and one.

- Checked the alignment of outer-test graph identities, labels and predictions before accepting recorded accuracy.

    - The original test_indices identified which dataset graphs belonged to the current outer-test fold and the order in which their predictions were recorded.

    - dataset[index].y.item() retrieved the processed class label for each of those original graph indices.

    - The resulting label list had to match the stored labels exactly, and predictions had to contain one valid class ID per label.

    - zip(predictions, labels) paired each prediction with its corresponding true class.

    - Summing the equality comparisons counted correctly classified graphs, and dividing by the number of labels recomputed accuracy.

    - math.isclose compared the recomputed value with the stored accuracy using an absolute tolerance of 1e-12 and no relative tolerance. This allowed negligible floating-point differences without accepting a materially different measurement.

- main collected all accepted groups before printing either summary table.

    - Parameter counts had to agree across the five fits within a dataset/model group because fold membership does not change that model's architecture.

    - The nested result lists were flattened into all_results for checks applying across the complete reference matrix.

    - Device information, PyTorch version, PyG version, CUDA build and runtime-convention text had to agree across the records before their training times were presented together.

    - first_result supplied the shared device and software values for display after agreement had been checked.

    - The source commits were collected into a set and sorted so every distinct recorded source revision was displayed once.

- Simplified the terminal presentation while retaining the existing validation and summary calculations.

    - The accuracy heading became Outer-test accuracy (%), with separate columns for each outer fold, Mean and Sample SD.

    - The resource heading became Parameters and training time, with explicit Total time (s) and Mean epoch time (s) columns.

    - Retained one short clarification: Training passes only; time totals cover all outer folds.

    - Removed the longer explanations of the timing calculations, the full runtime-convention paragraph and the generic evidence-path template from terminal output.

    - Detailed explanations remained in the log, and the complete runtime convention remained in each JSON record.

- Ran python -m experiments.summarise again after simplifying the output.

    - The opening counts were 75 validated reference fits, 15 dataset/model groups and five outer folds per group.

    - All records reported source commit c60bbd57fd1c7cc7e6bce3a4a51c6fdc54b0776f.

    - The recorded device was the NVIDIA GeForce RTX 4070 Laptop GPU.

    - The recorded software was PyTorch 2.13.0+cu130, PyG 2.8.0.post1 and CUDA build 13.0.

    - Execution printed both complete tables and returned to PowerShell without a reported error.

    - The displayed fold accuracies, means, sample standard deviations, parameter counts and training times matched the earlier summary. The revision changed presentation rather than the numerical results.

- The accuracy table retained the five individual outer-test accuracies before calculating their mean and sample standard deviation.

    - Each value described the validation-selected model's performance on its own outer-test graphs.

    - np.array collected the five recorded accuracy proportions, and multiplication by 100 converted them to percentages.

    - accuracies.mean() calculated their unweighted arithmetic mean, giving each outer fold equal weight.

    - For five percentage accuracies a_0 through a_4, the mean was (a_0 + a_1 + a_2 + a_3 + a_4) / 5.

    - This differs from pooling all correct predictions and dividing by the total number of graphs when outer-fold sizes differ. The declared summary used the mean of fold accuracies.

- accuracies.std(ddof=1) calculated the sample standard deviation across the five outer folds.

    - NumPy's ddof argument reduces the variance denominator by the specified amount. With five values and ddof=1, the denominator was 5 - 1 = 4.

    - The calculation was the square root of the sum of squared deviations from the fold mean divided by four.

    - Because the input accuracies were percentages, the resulting standard deviation was expressed in percentage points.

    - A larger value indicated greater variation among the five observed fold accuracies.

    - This was descriptive fold variability, not a confidence interval or a measurement of variability from repeated training seeds on the same fixed split. The outer-fold fits also shared overlapping fitting data.

- The GCN/MUTAG row illustrated the accuracy calculation.

    - Outer folds 0 through 4 displayed accuracies of 78.95%, 81.58%, 86.84%, 64.86% and 75.68%.

    - Their reported mean was 77.58%, with sample standard deviation 8.20 percentage points.

    - The five values showed that the mean represented materially different fold outcomes rather than every fold achieving approximately the same score.

    - MUTAG outer-test folds contained 37 or 38 graphs, so one additional correct prediction changed an individual fold's accuracy by approximately 2.6 to 2.7 percentage points.

    - Calculations used the stored unrounded accuracies. The .2f formatting rounded only the displayed fold values, means and standard deviations.

- The MUTAG reference summaries were recorded as mean accuracy followed by sample standard deviation.

    - GCN: 77.58%, with SD 8.20 percentage points.

    - GraphSAGE: 79.73%, with SD 6.10 percentage points.

    - GIN: 81.35%, with SD 7.43 percentage points.

    - GAT: 79.23%, with SD 5.19 percentage points.

    - GATv2: 81.34%, with SD 6.98 percentage points.

    - GIN and GATv2 had very close observed means. Their difference did not establish a meaningful performance advantage.

- The PROTEINS reference summaries were also dataset-dependent.

    - GCN: 74.57%, with SD 2.73 percentage points.

    - GraphSAGE: 73.67%, with SD 3.17 percentage points.

    - GIN: 72.59%, with SD 3.28 percentage points.

    - GAT: 74.12%, with SD 2.37 percentage points.

    - GATv2: 73.22%, with SD 2.78 percentage points.

    - GCN had the highest observed mean on this dataset, so the attention models did not uniformly exceed the contextual message-passing references.

- The NCI1 reference summaries differed again.

    - GCN: 71.41%, with SD 0.67 percentage points.

    - GraphSAGE: 73.38%, with SD 1.26 percentage points.

    - GIN: 73.38%, with SD 1.79 percentage points.

    - GAT: 72.31%, with SD 1.18 percentage points.

    - GATv2: 71.97%, with SD 1.38 percentage points.

    - GraphSAGE and GIN shared the highest displayed mean, but equality after rounding did not establish identical underlying accuracies or fold behaviour.

    - Every model had lower observed fold SD on NCI1 than on MUTAG. This described these particular results rather than establishing a general relationship between dataset size and model stability.

- The parameter table reported model size per fit rather than adding parameter counts across the five separately fitted models.

    - Total parameters included every registered model parameter, while trainable parameters included those eligible for optimisation.

    - The two counts were equal for every reference group.

    - For example, GCN/MUTAG reported 4,802 total and trainable parameters per model. The table did not multiply this by five.

    - GAT and GATv2 on NCI1 reported 6,978 and 13,570 parameters respectively.

    - Their shared representation width therefore did not imply identical parameter counts. The standard attention formulations retained their different parameterisations.

- The Total time (s) column added the recorded training-pass seconds across the five fits in each group.

    - The GCN/MUTAG group total was 65.04 seconds.

    - This covered all 500 training passes of each fit, including epochs after the eventually selected checkpoint.

    - Each group completed 5 × 500 = 2,500 training epochs.

    - The Mean epoch time (s) column divided summed training seconds by summed completed epochs. For GCN/MUTAG, 65.04 / 2,500 gave approximately 0.0260 seconds per training epoch.

    - This was an average training-pass duration across the group, not the time for a minibatch or a complete training-and-validation cycle.

    - The recorded mean_seconds_per_epoch for each individual fit was also checked against its training_seconds divided by completed_epochs before the group summary was calculated.

- The stored runtime convention defined the boundaries of those timing measurements.

    - Included operations were training-loader iteration, device transfer, forward computation, loss calculation, backpropagation, optimiser updates and training metric calculation and accumulation.

    - Excluded operations were validation, outer-test evaluation, checkpoint copying and restoration, progress printing, model-state saving and JSON writing.

    - Total time (s) therefore did not represent the complete elapsed time spent waiting for a dataset/model group to finish.

    - The summariser continued checking that the recorded runtime-convention descriptions agreed, even though the full description was no longer printed.

- Recorded training cost increased substantially across the displayed dataset groups.

    - GCN training-pass totals were 65.04 seconds on MUTAG, 302.54 seconds on PROTEINS and 1,106.23 seconds on NCI1.

    - Their corresponding mean training-pass durations were 0.0260, 0.1210 and 0.4425 seconds per epoch.

    - GATv2 had higher recorded training-pass totals than GAT on all three datasets.

    - On NCI1, GAT recorded 1,393.68 seconds across the five fits and GATv2 recorded 1,427.94 seconds, corresponding to 0.5575 and 0.5712 seconds per training epoch.

    - These measurements described the recorded hardware, software and workloads. They were not treated as hardware-independent operator costs.

- The summaries remained reproducible from the preserved reference JSON records.

    - For example, results/cross_validation_gcn_mutag_fold0_seed2000.json identified the GCN/MUTAG record for outer fold 0 and training seed 2000.

    - The associated .pt file preserved that fit's selected model state, while the JSON supplied the values used by the summariser.

    - Both tables were printed to the terminal and could be regenerated by running python -m experiments.summarise.

- The reference results showed no model leading consistently across all three datasets.

    - GATv2 had a higher mean than GAT on MUTAG and lower means on PROTEINS and NCI1.

    - These were descriptive observations under the fixed project pipeline, not significance tests, equivalence claims or reasons to change the experimental settings.

    - Validation losses, selected epochs, partitions and predictions supported record validation, while the displayed summaries focused on outer-test accuracy, model size and training cost.

    - The completed reference records remained the evidence source for the later GAT/GATv2 comparison, eight-head comparison and fitted reference-GAT analysis.

- Commit: 6.6 added validated reference result summaries





# Stage 6 Closing Notes

- Decisions

    - Completed the dataset scope with MUTAG, PROTEINS and NCI1. All four agreed research questions use these same datasets so comparisons retain a common assessment basis.

    - Retained the established input policy: categorical node information and graph connectivity are supplied to the models, while continuous node attributes and edge features are excluded. Dataset-specific input widths remain appropriate because the categorical vocabularies differ.

    - Fixed the investigation at four questions: first-layer attention-head count, standard GAT versus standard GATv2, uniform-attention retraining, and fitted reference-GAT attention characterisation with a uniform-attention intervention.

    - Retained the shared two-layer graph-classification structure, hidden width 64, ReLU after both complete convolutions, sum pooling and linear graph classifier. Adam uses learning rate 0.01, weight decay 0.0005 and batch size 32.

    - Fixed the common assessment allowance at 500 epochs before any cross-validation results were available. The development-validation convergence audit supported this as a practical computational compromise; it did not establish that every configuration would reach its eventual minimum validation loss within 500 epochs.

    - Adopted five stratified outer folds, with one fresh fit per model and outer fold. Each outer remainder is divided into fitting and validation partitions, giving approximately 70/10/20 of the complete dataset for fitting, validation and outer-test assessment.

    - Fixed outer split seed 0, validation split seed 1000 + fold_id and training seed 2000 + fold_id. Corresponding models and variants use the same partition indices and scheduled seeds.

    - Retained validation-loss checkpoint selection without early stopping. Every valid fit completes all 500 epochs, preserves the earliest state attaining the minimum validation cross-entropy, restores that state and evaluates the outer-test partition once.

    - Added finite-loss guards so numerical failure stops a fit before it can be accepted as a completed reference result. An earlier valid checkpoint does not justify recording a run containing non-finite training or validation losses.

    - Made experiments/train_cv.py the configuration owner and reference-training entry point. experiments/ablation.py derives its settings from that runner and uses the same cross-validation function.

    - Preserved superseded implementations in the mirrored archive/src and archive/experiments directories. The active evaluator returns loss, accuracy, predictions and labels; the original two-value evaluator remains archived.

    - Retained scientific qualifiers such as outer_folds, outer_split_seed, validation_split_seed and selected_epoch. The outer assessment remains an outer assessment when validation uses a single holdout rather than inner cross-validation.

    - Required one JSON evidence record and one paired validation-selected model state for every recorded reference fit. Existing output paths are rejected before fitting the corresponding dataset/model group, preventing silent replacement of recorded evidence.

    - Added reference summarisation only after the records existed. experiments/summarise.py uses os.listdir and filename checks, validates complete groups and calculates the declared statistics from preserved measurements.

    - Kept terminal output focused on identities, settings, measurements and brief clarifications. Explanations of calculations, methodology and interpretation belong in the logs. The simplified summary output retained the same numerical results.

    - Preserved the fixed settings after observing the reference results. Dataset-dependent model ordering did not trigger another search, selective reruns or changes to the agreed comparisons.

- Ideas

    - Carried forward the agreed investigation of what the fitted reference GAT actually learns about neighbourhood weighting. The analysis will use the selected reference states and their corresponding outer-test graphs, keeping each graph's assessment out of fold.

    - The agreed characterisation will measure departure from uniform attention separately for both GAT layers, with receiver-level measurements aggregated within graphs and graphs weighted equally within each fold.

    - The accompanying fitted intervention will replace learned attention scoring with uniform weighting while retaining the fitted message transformations and classifier. This will measure prediction sensitivity without retraining and remains distinct from training a fresh uniform-attention model.

    - These investigations are already part of the four-question scope. No further experiment was adopted in response to the observed reference rankings.

- Report notes

    - PROTEINS contains 1,113 graphs and three loaded categorical node-feature channels. One continuous node attribute is available but excluded by the project policy. NCI1 contains 4,110 graphs and 37 categorical node-feature channels, with no continuous node attributes. Both datasets have two processed graph classes and no loaded edge features.

    - The ordinary GCN development fits established that the shared loading, batching, training and selected-state recording path operated on PROTEINS and NCI1. Their approximately 80/10/10 development results remain separate from the later five-fold assessment results.

    - The epoch-budget audit recorded 27 development-validation trajectories covering nine configurations on three datasets. In the clean rerun, 16 trajectories had reached their eventual minimum validation loss by epoch 500. The median remaining loss gap was zero, the mean was approximately 0.00476 and the maximum was approximately 0.06177.

    - The clean audit recorded 7,928.40 elapsed seconds, approximately 2.20 hours, in results/development_epoch_budget_audit.json. The earlier interrupted audit remains separate historical evidence and is not the source of these clean-run measurements.

    - The epoch allowance was informed by development validation on the same benchmark datasets later used for cross-validation. It was fixed before CV outcomes and was not chosen using development-test scores. Methods must describe this sequence, and Limitations must acknowledge that the earlier benchmark use prevents claiming complete assessment independence from the budget decision.

    - Within each CV fit, fitting, validation and outer-test indices are disjoint. Validation chooses the checkpoint, while the outer-test partition measures the selected model's held-out performance. This within-fold separation does not remove the earlier dataset reuse in the budget decision.

    - The reference matrix completed 75 fits: five models on three datasets across five outer folds. The repository contains 75 corresponding JSON records and 75 paired model-state files.

    - All reference records reported source commit c60bbd57fd1c7cc7e6bce3a4a51c6fdc54b0776f. Their recorded environment used the NVIDIA GeForce RTX 4070 Laptop GPU, PyTorch 2.13.0+cu130, PyG 2.8.0.post1 and CUDA build 13.0.

    - experiments/summarise.py accepted all 15 complete reference groups. Its checks covered effective settings, scheduled seeds, expected partitions, original-index label alignment, prediction-derived accuracy, selection metadata, parameter-count consistency, paired state-file existence and recorded runtime conditions.

    - Reported accuracy is the unweighted mean of five outer-fold accuracies, accompanied by sample standard deviation calculated with ddof=1. Accuracy is expressed as a percentage and SD in percentage points. The individual fold values remain available for matched comparisons.

    - Fold SD describes variation in the observed five-fold procedure. It is not a confidence interval, a significance test or an estimate obtained from repeated training seeds on an unchanged split. The outer-fold fits also share overlapping fitting data.

    - The reference results showed dataset-dependent model ordering. GIN and GATv2 had close MUTAG means of 81.35% and 81.34%. GCN had the highest PROTEINS mean at 74.57%. GraphSAGE and GIN both displayed a NCI1 mean of 73.38%, without establishing identical underlying fold behaviour.

    - GATv2 had a higher mean than GAT on MUTAG and lower means on PROTEINS and NCI1. The comparison does not establish a universal advantage for either formulation or isolate query-dependent attention from all other parameterisation differences.

    - MUTAG had the largest observed fold SD for every reference model. Its outer-test folds contain only 37 or 38 graphs, so a single correct prediction changes a fold's accuracy by approximately 2.6 to 2.7 percentage points.

    - Parameter counts are reported per fitted model, with total and trainable counts distinguished. All reference-model parameters were trainable. The distinction remains necessary for the later uniform-attention model because its scoring parameters are frozen.

    - Runtime summaries use training-pass seconds, covering minibatch loading, device transfer, forward computation, loss, backpropagation, optimisation and training metric accumulation. Validation, outer testing, checkpoint copying and restoration, printing and file saving are excluded.

    - Each reference group completed 2,500 training epochs across its five fits. Total time (s) sums their training-pass times, while Mean epoch time (s) divides that total by their completed epochs. These quantities are distinct from complete program elapsed time.

    - GATv2 had more parameters and higher recorded training-pass totals than GAT on all three datasets. These observations apply to the recorded implementation, hardware and workloads.

    - The complete fold values, means, sample SDs, parameter counts and training times can be regenerated with python -m experiments.summarise from the preserved cross_validation JSON records in results/.

    - The eight-head GAT and GATv2 reference groups will be reused for the relevant Stage 7 comparisons. The remaining optimisation work is 45 fits for the additional one-, two- and four-head GAT variants and 15 fits for uniform GAT, bringing the planned total to 135 fits.

    - The fitted-attention characterisation and intervention reuse the 15 selected reference-GAT states and require no additional optimisation fits. Those analyses and the remaining 60 training fits have not yet been executed.

    - Stage 6 established the reference evidence and the common procedure for the attention investigation. Substantive manuscript drafting remains for the dedicated report phase.

- Commit: Stage 6 recorded closing decisions and report notes