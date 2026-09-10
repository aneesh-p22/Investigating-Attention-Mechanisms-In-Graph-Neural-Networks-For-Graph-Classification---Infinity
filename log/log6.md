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