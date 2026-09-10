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