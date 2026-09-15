# 7.1 Fixed-Width Attention-Head Experiment

- Investigated RQ1 by varying the number of first-layer GAT attention heads while keeping total first-layer representation width fixed at 64.

    - Compared 1, 2, 4 and 8 heads on MUTAG, PROTEINS and NCI1.

    - One head used 64 channels per head.

    - Two heads used 32 channels per head.

    - Four heads used 16 channels per head.

    - Eight heads used 8 channels per head.

    - Multiplying head count by channels per head therefore gave first-layer width 64 in every configuration.

    - The second GAT layer remained one 64-channel head with concat=False.

    - All other architecture and assessment settings remained unchanged.

- The fixed-width design separates head count from simply increasing model size.

    - More heads provide more independently learned attention distributions.

    - Under fixed total width, however, each individual head becomes narrower as head count increases.

    - The comparison therefore studies how one fixed representation is divided between attention heads rather than giving larger-head configurations a wider representation.

    - There was no assumption that increasing head count should monotonically improve classification accuracy.

- Reused the established Stage 6 cross-validation procedure for all new fits.

    - experiments/train_cv.py remained responsible for dataset loading, fold reconstruction, fitting and validation partitions, model construction, optimisation, checkpoint selection, outer-test evaluation and result recording.

    - Adam retained learning rate 0.01 and weight decay 0.0005.

    - Batch size remained 32 and every valid fit completed 500 epochs.

    - Outer split seed remained 0.

    - Validation split seed remained 1000 + fold_id.

    - Training seed remained 2000 + fold_id.

    - The earliest state attaining minimum validation cross-entropy was restored before outer-test evaluation.

- Reorganised experiments/ablation.py so the three RQ1 configurations requiring new optimisation are collected in head_variants.

    - head_variants contains the existing heads1, heads2 and heads4 GAT definitions.

    - Their scientific settings were not changed.

    - The complete variants collection still also contains the eight-head reference GAT, reference GATv2 and Uniform GAT.

    - The eight-head GAT and GATv2 remain marked for reference reuse.

    - Uniform GAT remains a separate future experiment rather than part of RQ1.

- Retained experiments/ablation.py as an inspection and configuration module.

    - python -m experiments.ablation still prints each attention configuration and its effective settings without starting optimisation.

    - get_variant_settings constructs a complete variant configuration from the shared assessment settings, normal model settings and variant-specific overrides.

    - run_variant applies a non-reference variant to all three datasets and sends it through run_cross_validation.

    - Reference variants are rejected by run_variant so they cannot accidentally be retrained through the ablation path.

- Added experiments/train_heads.py as the explicit execution entry point for Stage 7.1.

    - The file imports head_variants and run_variant rather than duplicating training logic.

    - Its main function loops only over heads 1, 2 and 4.

    - Each head count is evaluated on three datasets and five outer folds.

    - This gives 3 × 3 × 5 = 45 new fits.

    - The existing eight-head reference GAT contributes another 15 previously completed fits to the final RQ1 comparison.

- Extended experiments/summarise.py so the existing validation machinery can also process named variants.

    - Reference results continue to use filenames such as cross_validation_gat_mutag_fold0_seed2000.json.

    - Variant results include the variant name, such as cross_validation_gat_mutag_heads1_fold0_seed2000.json.

    - load_group now adds the variant component to its filename prefix when variant is not None.

    - The established settings, partition, state-file, prediction, loss, parameter and runtime checks therefore apply to both reference and head-count records.

- Reused common summary calculations instead of creating separate RQ1 implementations.

    - check_parameter_counts verifies that all five folds within one configuration report the same parameter counts.

    - check_runtime_conditions applies the existing device, software and runtime-convention agreement checks.

    - print_accuracy_table calculates fold percentages, arithmetic mean and sample standard deviation for both the reference and RQ1 sections.

    - print_resource_table performs the corresponding parameter and training-time summary.

    - The Stage 6 reference calculations therefore remain the same while being reused for the new evidence.

- Reused the existing eight-head reference GAT results rather than fitting another eight-head configuration.

    - The already validated reference GAT groups were retained while the reference matrix was loaded.

    - These five-fold groups supply the eight-head condition for each dataset.

    - Heads 1, 2 and 4 use the newly generated variant records.

    - The final RQ1 evidence therefore contains 45 new fits and 15 reused reference fits.

- Added an explicit parameter-count check across heads 1, 2, 4 and 8 within each dataset.

    - MUTAG reported 5,058 total and trainable parameters for every head count.

    - PROTEINS reported 4,802 total and trainable parameters for every head count.

    - NCI1 reported 6,978 total and trainable parameters for every head count.

    - The fixed-width comparison therefore also preserved overall model parameter count within each dataset.

- Ran python -m experiments.ablation before fitting.

    - Heads 1, 2 and 4 were reported as additional cross-validation fits.

    - Their derived channels per head were 64, 32 and 16 respectively.

    - The reused eight-head GAT used 8 channels per head.

    - All four configurations reported first-layer width 64 and embedding width 64.

    - GATv2 remained a reused reference configuration.

    - Uniform GAT remained a separate additional-fit configuration.

    - The printed example result paths matched the planned heads1, heads2 and heads4 naming scheme.

- Committed the Stage 7.1 source before producing the final experimental evidence.

    - The new head-count records therefore identify source commit 59b4777610a56b06fb9dec3c3c2da8b9416791b0.

    - The reused eight-head reference records retain their original source commit c60bbd57fd1c7cc7e6bce3a4a51c6fdc54b0776f.

    - Keeping both hashes preserves the actual execution provenance of the new and reused evidence.

- The first execution of python -m experiments.train_heads was interrupted by an unexpected laptop restart.

    - Five one-head MUTAG folds, five one-head PROTEINS folds and two one-head NCI1 folds had completed before the restart.

    - These 12 completed fits had produced 24 untracked files, one JSON result and one PT model state per fit.

    - A filename filter was used to preview only the Stage 7.1 heads1, heads2 and heads4 result and state files.

    - The partial Stage 7.1 files were then removed.

    - Existing reference results were not removed.

    - git status returned to the clean committed pre-run state.

    - The full 45-fit experiment was restarted from the beginning using the same committed code and predetermined seeds.

    - No output from the interrupted attempt was retained as final evidence.

- The restarted execution completed all 45 required new fits.

    - Every heads1, heads2 and heads4 dataset/fold combination completed the full 500-epoch allowance.

    - Each fit restored its selected validation state before outer-test evaluation.

    - Each completed fit saved one JSON result and one paired PT model-state file.

    - The final scheduled fit was the four-head GAT on NCI1 outer fold 4.

    - It completed normally and returned control to PowerShell without a reported error.

- Ran python -m experiments.summarise after all new fits were complete.

    - The reference section again validated 75 reference fits across 15 dataset/model groups.

    - The RQ1 section validated 60 contributing fits.

    - These comprised 45 new heads1, heads2 and heads4 fits and 15 reused eight-head reference GAT fits.

    - Every dataset/head combination contained all five expected outer folds.

    - Runtime conditions were consistent across the contributing records.

    - The recorded device was the NVIDIA GeForce RTX 4070 Laptop GPU.

    - The recorded software was PyTorch 2.13.0+cu130, PyG 2.8.0.post1 and CUDA build 13.0.

- MUTAG produced the following fixed-width head-count results.

    - One head: 76.51% mean outer-test accuracy, sample SD 7.24 percentage points.

    - Two heads: 77.07% mean, sample SD 5.50 percentage points.

    - Four heads: 75.48% mean, sample SD 5.10 percentage points.

    - Eight heads: 79.23% mean, sample SD 5.19 percentage points.

    - Eight heads had the highest observed mean.

    - The ordering was not monotonic because performance increased from one to two heads, decreased at four heads and then increased at eight heads.

    - Fold-level winners also differed, with one head leading outer fold 0, two heads leading outer fold 1 and eight heads leading outer folds 2 through 4.

- PROTEINS produced much closer head-count means.

    - One head: 73.94% mean, sample SD 2.82 percentage points.

    - Two heads: 73.40% mean, sample SD 1.89 percentage points.

    - Four heads: 73.94% mean, sample SD 2.42 percentage points.

    - Eight heads: 74.12% mean, sample SD 2.37 percentage points.

    - The complete range between the highest and lowest observed means was only 0.72 percentage points.

    - Different outer folds favoured different head counts.

    - The results therefore provided little descriptive evidence for a strong head-count advantage on PROTEINS.

- NCI1 produced a different ordering again.

    - One head: 71.31% mean, sample SD 1.39 percentage points.

    - Two heads: 72.51% mean, sample SD 1.02 percentage points.

    - Four heads: 72.09% mean, sample SD 1.05 percentage points.

    - Eight heads: 72.31% mean, sample SD 1.18 percentage points.

    - Two heads had the highest observed mean.

    - Its observed advantage over eight heads was only 0.20 percentage points.

    - The relationship was again non-monotonic.

- The three datasets therefore did not support a general rule that increasing attention-head count improves graph-classification accuracy.

    - Eight heads had the highest observed mean on MUTAG and PROTEINS.

    - Two heads had the highest observed mean on NCI1.

    - No head count was best on every dataset or every outer fold.

    - The observed effect of head count was therefore dataset-dependent under the fixed assessment pipeline.

- The result is consistent with head count representing a trade-off rather than a simple scale parameter.

    - More heads provide more separately learned attention distributions.

    - Fixed total width means that each additional head also reduces the number of channels available to an individual head.

    - Different heads can also learn partly redundant neighbour-weighting patterns.

    - Increasing head count therefore does not guarantee additional useful representation capacity.

    - The most useful balance between attention diversity and per-head width can depend on the dataset.

- Recorded training-pass times were similar but not identical across head counts within each dataset.

    - MUTAG totals for heads 1, 2, 4 and 8 were 76.50, 76.38, 76.01 and 78.22 seconds.

    - PROTEINS totals were 367.97, 368.03, 374.22 and 375.58 seconds.

    - NCI1 totals were 1,342.07, 1,335.47, 1,343.98 and 1,393.68 seconds.

    - Eight heads had the largest recorded training-pass total on all three datasets.

    - Equal parameter counts therefore do not imply identical execution time.

    - These values describe the recorded hardware and software environment rather than hardware-independent computational complexity.

- Interpreted the RQ1 evidence descriptively rather than inferentially.

    - Each mean summarises five outer-fold accuracies.

    - The sample standard deviation describes variation among those five observed fold accuracies.

    - It is not a confidence interval.

    - The folds are not repeated training-seed runs on one fixed split.

    - Their fitting sets also overlap as part of the established outer cross-validation procedure.

    - Close observed means therefore do not establish statistical equivalence.

- The RQ1 results were not used to alter the predetermined reference configuration.

    - No head count was rerun because its observed performance appeared poor.

    - No dataset-specific head count was substituted for the existing reference GAT.

    - The eight-head reference remains the configuration fixed before final assessment.

    - Stage 7.1 is therefore an analysis of head-count behaviour rather than another hyperparameter-selection stage.

- Commit: 7.1 recorded the fixed-width attention-head study





# 7.2 Matched GAT and GATv2 Comparison

- Investigated RQ2 by comparing the existing reference GAT and GATv2 results under the common Stage 6 assessment pipeline.

    - No new model fitting was required.

    - The comparison reused the five reference GAT folds and five reference GATv2 folds for MUTAG, PROTEINS and NCI1.

    - This gave 30 reused fits in total.

    - All contributing records retained source commit c60bbd57fd1c7cc7e6bce3a4a51c6fdc54b0776f from the original reference execution.

- The comparison retained the common project settings while preserving the intended architectural difference between standard GAT and standard general-form GATv2.

    - Both models used hidden width 64 and eight first-layer attention heads.

    - Both used one 64-channel second-layer head, ReLU, global sum pooling and the final linear classifier.

    - They also shared the same datasets, feature policy, outer folds, validation split schedule, training seed schedule, optimiser settings, batch size, 500-epoch allowance and checkpoint-selection procedure.

    - GATv2 retained share_weights=False.

    - Parameter counts were therefore not artificially forced to match.

- Extended experiments/summarise.py to retain both validated reference GAT and GATv2 groups.

    - reference_gat_groups stores the five accepted GAT folds for each dataset.

    - reference_gatv2_groups stores the corresponding five accepted GATv2 folds.

    - These records had already passed the established settings, partition, prediction, state-file, parameter and runtime checks before entering the RQ2 comparison.

- Added a matched fold comparison rather than comparing only the two overall means.

    - get_paired_accuracy_differences pairs the GAT and GATv2 result with the same outer-fold ID.

    - It checks that paired records contain the same saved partition dictionary.

    - It also checks that their ordered outer-test labels agree.

    - This confirms that each accuracy difference compares predictions on the same held-out graphs.

    - The difference for one fold is calculated as GATv2 outer-test accuracy minus GAT outer-test accuracy and expressed in percentage points.

    - A positive value therefore favours GATv2 and a negative value favours GAT.

- Added print_gat_gatv2_accuracy_comparison to display the matched RQ2 evidence.

    - The table reports the GAT and GATv2 mean accuracies for each dataset.

    - It then reports all five matched outer-fold accuracy differences.

    - Their arithmetic mean gives the average paired difference.

    - Their sample standard deviation describes variation among the five observed fold differences.

    - These paired differences retain information that would be hidden by comparing only the two model means.

- Added a separate parameter-count comparison.

    - This reports the total and trainable parameter counts of GAT and GATv2 for each dataset.

    - It also reports the difference GATv2 minus GAT.

    - The comparison records the actual capacity difference introduced by the chosen standard general-form GATv2 configuration rather than treating the models as parameter matched.

- Ran python -m experiments.summarise after adding the RQ2 section.

    - The existing reference section again validated all 75 reference fits.

    - The RQ1 section again validated its 60 contributing fits.

    - The new RQ2 section validated 30 reused reference fits.

    - The new-fit count was correctly reported as 0.

    - The command returned to PowerShell without a reported error.

- MUTAG was the only dataset on which GATv2 had the higher observed mean.

    - GAT mean outer-test accuracy was 79.23%.

    - GATv2 mean outer-test accuracy was 81.34%.

    - The matched fold differences were +7.89, 0.00, +2.63, -5.41 and +5.41 percentage points.

    - Their mean was +2.11 percentage points.

    - Their sample standard deviation was 5.14 percentage points.

    - GATv2 therefore performed better on average on MUTAG, but the direction and size of the difference varied substantially between outer folds.

- PROTEINS favoured GAT on average.

    - GAT mean outer-test accuracy was 74.12%.

    - GATv2 mean outer-test accuracy was 73.22%.

    - The matched fold differences were -1.35, +0.45, -0.90, -1.35 and -1.35 percentage points.

    - Their mean was -0.90 percentage points.

    - Their sample standard deviation was 0.78 percentage points.

    - Four of the five folds favoured GAT, while one slightly favoured GATv2.

- NCI1 also favoured GAT on average, although the observed mean difference was small.

    - GAT mean outer-test accuracy was 72.31%.

    - GATv2 mean outer-test accuracy was 71.97%.

    - The matched fold differences were +0.85, 0.00, -0.61, -1.70 and -0.24 percentage points.

    - Their mean was -0.34 percentage points.

    - Their sample standard deviation was 0.93 percentage points.

    - The fold-level direction was therefore mixed and the overall observed separation was small.

- GATv2 contained substantially more parameters than GAT under the chosen general-form configuration.

    - On MUTAG, GAT contained 5,058 parameters and GATv2 contained 9,730, a difference of 4,672.

    - On PROTEINS, GAT contained 4,802 parameters and GATv2 contained 9,218, a difference of 4,416.

    - On NCI1, GAT contained 6,978 parameters and GATv2 contained 13,570, a difference of 6,592.

    - All recorded parameters were trainable.

    - The comparison is therefore between the standard project GAT and standard general-form GATv2 rather than two parameter-matched architectures.

- Interpreted the results in relation to the static and dynamic attention distinction studied in Stage 5.

    - Standard GAT has the static-ranking restriction within each attention head.

    - GATv2 changes the scoring formulation so that receiver-dependent sender rankings can be represented.

    - This gives GATv2 a more expressive attention mechanism.

    - Greater attention expressivity does not guarantee better graph-classification accuracy on every dataset.

    - A dataset may not require the additional receiver-dependent ranking capability, and additional model flexibility can interact with optimisation and generalisation rather than producing an automatic improvement.

- RQ2 did not show a consistent accuracy advantage for GATv2.

    - GATv2 had the higher observed mean on MUTAG.

    - GAT had the higher observed mean on PROTEINS and NCI1.

    - The MUTAG advantage also varied considerably across outer folds.

    - The results therefore do not support a general claim that replacing standard GAT attention with the more expressive GATv2 mechanism improves graph-classification performance under this pipeline.

    - They instead show dataset-dependent behaviour despite the theoretical expressivity advantage of dynamic attention.

- The RQ2 conclusion remains descriptive.

    - The five differences come from the five established outer folds rather than repeated training seeds on one fixed partition.

    - Their sample standard deviation is not a confidence interval.

    - The fitting sets overlap across outer folds.

    - No statistical significance or equivalence claim is made.

    - The parameter-count difference also means the observed accuracy differences cannot be attributed solely to static versus dynamic attention in isolation.

- Commit: 7.2 analysed the matched GAT and GATv2 results





# 7.3 Uniform-Attention Retraining

- Investigated RQ3 by comparing the fitted reference GAT with fresh GAT models trained under the established uniform-attention constraint.

    - The learned GAT results were reused from the completed reference cross-validation experiment.

    - One fresh Uniform GAT was trained for each of the five outer folds on MUTAG, PROTEINS and NCI1.

    - This produced 15 new fits.

    - Together with the 15 reused learned-GAT fits, the final RQ3 comparison contained 30 validated result records.

- Kept the established reference architecture and assessment protocol while changing whether the attention-scoring mechanism could learn non-uniform coefficients.

    - Both models retained hidden width 64.

    - Both used eight first-layer heads with eight channels per head and one 64-channel second-layer head.

    - Both retained the same message transformations, biases, ReLU activations, self-loop handling, global sum readout and linear graph classifier.

    - Both used the same datasets, feature policy, folds, validation partitions, training seeds, optimiser settings, batch size, 500-epoch training allowance and validation-loss checkpoint-selection rule.

    - Uniform GAT additionally fixed uniform_attention=True.

- Retained the uniform-attention implementation established before final assessment.

    - conv1.att_src, conv1.att_dst, conv2.att_src and conv2.att_dst are set to zero for a fresh Uniform GAT.

    - Their requires_grad flags are False before the optimiser is created.

    - Equal attention logits therefore produce equal softmax coefficients over the eligible incoming entries for each receiver and head.

    - The message transformations, biases and classifier remain trainable.

    - The graph connectivity and inserted self-loops remain part of message passing.

- Reused the existing integrated Uniform GAT inspection before starting the final RQ3 fits.

    - The MUTAG control model contained 5,058 total parameters and 4,802 trainable parameters.

    - Exactly 256 parameters were frozen.

    - The four attention-scoring tensors were all zero and non-trainable.

    - For the inspected receiving node with three incoming entries, every first-layer head assigned coefficient 0.3333 to each entry.

    - The second layer also assigned coefficient 0.3333 to each of the three entries.

    - Incoming coefficients summed to one in every inspected head.

    - After one optimiser update, all four scoring tensors remained unchanged and zero.

    - The first message transformation and classifier changed, confirming that the remaining model still learned.

- Grouped the small experiment-specific training entry points under experiments/runners.

    - experiments/train_heads.py was moved to experiments/runners/train_heads.py.

    - experiments/runners/train_uniform.py was added for the RQ3 execution.

    - An empty experiments/runners/__init__.py makes the directory an explicit package.

    - The runner functions were named train_heads and train_uniform rather than using generic main names.

    - experiments/train_cv.py remained at the top level because it owns the reusable cross-validation procedure rather than selecting one particular experiment.

- Kept the Uniform GAT definition in experiments/ablation.py rather than duplicating its settings in the new runner.

    - The existing Uniform GAT dictionary was assigned the name uniform_variant.

    - train_uniform imports this variant and passes it to the established run_variant function.

    - run_variant continues to reuse run_cross_validation, so RQ3 does not introduce a second training implementation.

- Created the pre-fit source commit before final RQ3 optimisation.

    - The committed source revision was f68e626a88c54c160ea47b1d17d1b9c7e710ec7f.

    - The new Uniform GAT records were therefore produced from a fixed source state.

    - The reused learned GAT reference records retain their original source revision c60bbd57fd1c7cc7e6bce3a4a51c6fdc54b0776f.

    - The two source hashes are intentionally different because the learned reference fits and later Uniform GAT fits were executed from different committed source states.

- Ran python -m experiments.runners.train_uniform for the final RQ3 experiment.

    - All five MUTAG folds completed.

    - All five PROTEINS folds completed.

    - All five NCI1 folds completed.

    - Each fit completed the full 500-epoch training allowance.

    - Each fit restored the state with minimum validation cross-entropy under the established earliest-exact-tie rule before outer-test assessment.

    - A paired PT model state and JSON result record was saved for every new fold.

    - The execution returned normally to PowerShell after NCI1 outer fold 4.

- Extended experiments/summarise.py to validate and compare the new Uniform GAT evidence.

    - Uniform result groups are loaded using the same settings, filename, state-file, dataset-policy, partition, prediction, loss, parameter and runtime checks used for the reference and earlier ablation results.

    - The Uniform GAT group for each dataset is paired with the existing reference GAT group.

    - Pairing checks require matching outer-fold IDs, partition dictionaries and ordered outer-test labels.

    - This ensures that each reported accuracy difference compares the two models on the same held-out graphs.

    - RQ3 differences are reported as learned GAT accuracy minus Uniform GAT accuracy in percentage points.

    - Positive values therefore favour learned attention and negative values favour Uniform GAT.

- Added explicit parameter-control checks for RQ3.

    - Learned and Uniform GAT must have the same total registered parameter count within each dataset.

    - The reference learned GAT must have all registered parameters trainable.

    - Uniform GAT must have fewer trainable parameters than total parameters because its attention-scoring tensors are frozen.

    - These checks passed for all three datasets.

- Ran python -m experiments.summarise after all 15 Uniform GAT fits were complete.

    - The existing reference section again validated 75 fits.

    - RQ1 again validated 60 contributing fits.

    - RQ2 again validated 30 reused fits.

    - RQ3 validated 30 contributing fits, consisting of 15 reused learned-GAT references and 15 new Uniform GAT fits.

    - The RQ3 source revisions were the expected learned-reference and Uniform-GAT commits.

- MUTAG favoured Uniform GAT in the observed outer-test results.

    - Learned GAT mean accuracy was 79.23%.

    - Uniform GAT mean accuracy was 80.83%.

    - The learned-minus-uniform fold differences were -2.63, -2.63, 0.00, -2.70 and 0.00 percentage points.

    - Their mean was -1.59 percentage points.

    - Their sample standard deviation was 1.45 percentage points.

    - Uniform GAT had higher accuracy on three folds and tied learned GAT on two.

    - The MUTAG evidence therefore does not indicate a performance benefit from learning non-uniform attention coefficients under this protocol.

- PROTEINS produced a small and fold-dependent observed advantage for learned GAT.

    - Learned GAT mean accuracy was 74.12%.

    - Uniform GAT mean accuracy was 73.58%.

    - The learned-minus-uniform fold differences were +0.45, +0.45, -1.79, +3.60 and 0.00 percentage points.

    - Their mean was +0.54 percentage points.

    - Their sample standard deviation was 1.95 percentage points.

    - Two folds favoured learned GAT, one favoured Uniform GAT, one was tied and one showed a larger learned-GAT advantage.

    - The direction was therefore mixed rather than consistently favouring learned attention.

- NCI1 showed the clearest observed advantage for learned non-uniform attention.

    - Learned GAT mean accuracy was 72.31%.

    - Uniform GAT mean accuracy was 71.09%.

    - The learned-minus-uniform fold differences were +0.36, +0.61, +1.95, +1.09 and +2.07 percentage points.

    - Their mean was +1.22 percentage points.

    - Their sample standard deviation was 0.77 percentage points.

    - All five outer folds favoured learned GAT.

    - This provides the most consistent RQ3 evidence that learned non-uniform weighting was useful on one of the adopted datasets.

- Total parameter counts were preserved between the two variants while the intended scoring parameters were removed from optimisation in Uniform GAT.

    - MUTAG contained 5,058 total parameters in both variants, with 5,058 trainable in learned GAT and 4,802 trainable in Uniform GAT.

    - PROTEINS contained 4,802 total parameters in both variants, with 4,802 trainable in learned GAT and 4,546 trainable in Uniform GAT.

    - NCI1 contained 6,978 total parameters in both variants, with 6,978 trainable in learned GAT and 6,722 trainable in Uniform GAT.

    - Uniform GAT therefore froze exactly 256 scoring parameters on every dataset.

    - The comparison deliberately does not equalise the number of trainable parameters because removing learnable attention scoring is the experimental constraint being studied.

- RQ3 does not support a universal claim that learning non-uniform neighbourhood coefficients improves graph-classification accuracy.

    - Uniform retraining achieved the higher observed mean on MUTAG.

    - PROTEINS showed a small mean advantage for learned attention with mixed fold directions.

    - NCI1 showed a modest learned-attention advantage consistently across all five folds.

    - The value of learning unequal attention coefficients was therefore dataset-dependent in this experiment.

- Kept the RQ3 inference boundary distinct from the upcoming fitted-model intervention.

    - RQ3 compares separately trained models that can adapt all parameters permitted by their respective constraints.

    - A Uniform GAT can therefore learn transformations and classifier parameters that compensate for its fixed neighbourhood weighting.

    - RQ3 does not directly establish how dependent an already fitted learned-attention GAT is on the particular non-uniform coefficients it learned.

    - Stage 7.4 addresses that different question by intervening on the preserved selected reference-GAT states without retraining their remaining parameters.

- The RQ3 evidence remains descriptive.

    - The five paired observations per dataset are established outer folds rather than repeated independent training seeds on one fixed split.

    - Outer-fold fitting sets overlap.

    - Sample standard deviation describes observed fold variation and is not a confidence interval or significance test.

    - No statistical significance or equivalence claim is made from these results.

- Commit: 7.3 recorded uniform-attention retraining results