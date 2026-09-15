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





# 7.4 Fitted Reference-GAT Attention Characterisation and Intervention

- Investigated RQ4 using the 15 validation-selected reference eight-head GAT states from MUTAG, PROTEINS and NCI1.

    - Each dataset contributed all five outer folds.

    - Every analysed graph belonged to the corresponding selected state's outer-test partition.

    - The five-fold rotation therefore analysed every graph out of fold once.

    - No new optimisation fits were performed.

    - The selected reference states retained source commit c60bbd57fd1c7cc7e6bce3a4a51c6fdc54b0776f.

- Separated RQ4 into fitted-attention characterisation and an inference-time uniform-attention intervention.

    - The characterisation measures how far the realised learned attention coefficients depart from uniform neighbourhood weighting.

    - The intervention measures how the same fitted model changes when its learned attention scoring is removed without retraining.

    - This differs from RQ3, where a separate Uniform GAT was trained from scratch and its remaining trainable parameters could adapt to the uniform-attention constraint.

- Used one minus normalised attention entropy as the predefined attention-departure measure.

    - For receiver i, head h and m_i greater than one effective incoming entries, normalised entropy is minus the sum of alpha log alpha divided by log(m_i).

    - Departure from uniformity is one minus this normalised entropy.

    - Perfectly uniform weighting gives departure zero.

    - Larger values indicate more concentrated and less uniform attention distributions.

    - Natural logarithms were used.

- Excluded receivers with only one effective incoming entry from the entropy calculation.

    - For m_i equal to one, log(m_i) is zero and there is no meaningful weighting choice.

    - These receivers were retained as counts rather than silently discarded.

    - MUTAG contained 3,371 eligible receivers and no single-entry receivers across its outer-test graphs.

    - PROTEINS contained 43,466 eligible receivers and 5 single-entry receivers.

    - NCI1 contained 122,319 eligible receivers and 428 single-entry receivers.

- Calculated the first-layer attention measure per head before averaging heads.

    - Conv1 has eight attention heads.

    - Departure was calculated separately for every head of one receiving node.

    - The eight departure values were then averaged to form the receiver-level Conv1 value.

    - Attention coefficients were not averaged across heads before entropy was calculated.

    - Conv2 contains one head and therefore required no head averaging.

- Aggregated the attention measure in the predefined hierarchy.

    - Eligible receiver values were averaged within each graph.

    - Graph values were averaged equally within each outer fold.

    - Conv1 and Conv2 remained separate throughout the analysis.

    - The five fold means for each dataset and layer were summarised with their arithmetic mean and sample standard deviation using ddof=1.

- Reused the cleaned shared result-validation support rather than duplicating cross-validation result checks inside the RQ4 script.

    - experiments/result_validation.py reconstructs the exact expected outer and validation partitions.

    - It opens the exact five expected result paths rather than scanning the result directory.

    - It checks settings, feature policy, partition indices, labels, predictions, recorded accuracy, losses, checkpoint-selection information, paired state paths, parameter counts, runtime information and source-commit consistency.

    - The cleaned experiments/summarise.py reproduced the previously accepted reference, RQ1, RQ2 and RQ3 output exactly before RQ4 was finalised.

    - The obsolete development comparison helper was moved from experiments/compare.py to archive/experiments/compare.py.

- Reconstructed two ordinary reference GATs for each selected state.

    - The exact same selected PT state was loaded into both copies.

    - The learned copy was left untouched.

    - Before intervention, every state tensor in the learned and intervention copies was required to be exactly equal.

    - This avoided introducing a separate intervention-specific GAT architecture.

- Required the untouched reconstructed model to reproduce its saved reference outer-test evaluation.

    - The recomputed ordered labels had to equal the reference JSON labels.

    - The recomputed predictions had to equal the stored predictions.

    - Recomputed accuracy had to agree exactly within the defined numerical tolerance.

    - Recomputed cross-entropy loss had to agree within the defined floating-point tolerance.

    - This connected every analysed selected state directly to its previously validated reference record.

- Implemented the fitted uniform-attention intervention by changing only the four attention-scoring tensors.

    - conv1.att_src was set to zero.

    - conv1.att_dst was set to zero.

    - conv2.att_src was set to zero.

    - conv2.att_dst was set to zero.

    - No optimiser was constructed and no retraining occurred.

    - Every non-attention-scoring state tensor was required to remain exactly equal to the untouched learned model.

    - Zero attention scorers give equal pre-softmax scores and therefore uniform coefficients over each receiver's effective incoming entries.

- Preserved graph-aligned evidence for the intervention.

    - Each saved graph record contains its graph ID and label.

    - It contains the learned prediction and intervention prediction.

    - It contains the Conv1 and Conv2 attention-departure values.

    - It also contains eligible-receiver and single-entry-receiver counts for both layers.

    - Fold records retain their fold ID, selected epoch and paired reference-model state path.

- Used three predefined fitted-intervention endpoints.

    - Delta loss is intervention loss minus learned loss.

    - Positive delta loss therefore means cross-entropy became worse after uniformising attention.

    - Delta accuracy is intervention accuracy minus learned accuracy.

    - Negative delta accuracy therefore means outer-test accuracy fell.

    - Prediction-flip rate is the fraction of held-out graphs whose predicted class changed under the intervention.

    - A model can change cross-entropy without changing its predicted class, so the three endpoints answer related but different questions.

- Added concise progress output before the final RQ4 execution.

    - The script reports the analysis source commit and device before analysis begins.

    - It reports each dataset, outer fold, selected epoch and number of outer-test graphs as the analysis proceeds.

    - This matches the existing project preference for long-running experimental commands to show meaningful execution progress without adding a progress-bar dependency or graph-by-graph output.

- An initial complete RQ4 execution used the same scientific analysis before the progress output was added.

    - That execution produced the same final numerical results.

    - Its result JSON was removed deliberately before the final rerun because the retained evidence should identify the exact final analysis source.

    - No reference model states or earlier experimental results were removed.

    - The progress-output change did not alter the scientific calculations.

- Committed the final RQ4 analysis source before the retained execution.

    - The final analysis source commit was f94c44c67f9d02bbaaf99f8f7bf01690f1c7de2c.

    - The reused reference states retained source commit c60bbd57fd1c7cc7e6bce3a4a51c6fdc54b0776f.

    - The working analysis therefore records separately the code that produced the reference fitted states and the code that analysed those states.

- Ran python -m experiments.analyse_fitted_gat for the final retained RQ4 evidence.

    - The command processed all 15 selected reference states.

    - MUTAG outer-test fold sizes were 38, 38, 38, 37 and 37 graphs, totalling all 188 graphs.

    - PROTEINS fold sizes were 223, 223, 223, 222 and 222, totalling all 1,113 graphs.

    - NCI1 used 822 test graphs in every fold, totalling all 4,110 graphs.

    - No optimisation fits were performed.

    - The analysis returned normally to PowerShell and saved results/rq4_fitted_gat_attention.json.

- The selected reference epochs were preserved and displayed during execution.

    - MUTAG selected epochs were 457, 439, 204, 475 and 16 for outer folds 0 through 4.

    - PROTEINS selected epochs were 187, 6, 5, 86 and 10.

    - NCI1 selected epochs were 256, 335, 178, 279 and 357.

    - These were the existing validation-selected reference states rather than newly selected RQ4 checkpoints.

- Conv1 attention was close to uniform on average on all three datasets.

    - MUTAG fold departures were 0.0122, 0.0074, 0.0020, 0.0217 and 0.0021.

    - MUTAG Conv1 mean departure was 0.0091 with sample SD 0.0082.

    - PROTEINS fold departures were 0.0198, 0.0045, 0.0032, 0.0119 and 0.0064.

    - PROTEINS Conv1 mean was 0.0092 with sample SD 0.0068.

    - NCI1 fold departures were 0.0119, 0.0359, 0.0355, 0.0315 and 0.0190.

    - NCI1 Conv1 mean was 0.0268 with sample SD 0.0108.

    - The first layer therefore showed only weak average departure from uniform weighting under this aggregate measure.

- Conv2 attention was much more dataset-dependent.

    - MUTAG fold departures were 0.0654, 0.1808, 0.2481, 0.0366 and 0.0518.

    - MUTAG Conv2 mean was 0.1165 with sample SD 0.0931.

    - PROTEINS fold departures were 0.0408, 0.0000, 0.0000, 0.0495 and 0.0000 at four-decimal reporting precision.

    - PROTEINS Conv2 mean was 0.0181 with sample SD 0.0249.

    - NCI1 fold departures were 0.3198, 0.3015, 0.2017, 0.3774 and 0.2105.

    - NCI1 Conv2 mean was 0.2822 with sample SD 0.0750.

    - NCI1 therefore showed the strongest and most consistently non-uniform second-layer attention among the three datasets.

- MUTAG showed modest and mixed sensitivity to the fitted uniform-attention intervention.

    - Delta-loss folds were -0.0037, 0.0012, 0.0424, 0.0970 and 0.0362.

    - Mean delta loss was 0.0346 with sample SD 0.0404.

    - Delta-accuracy folds were +5.26, 0.00, -2.63, 0.00 and -8.11 percentage points.

    - Mean delta accuracy was -1.10 percentage points with sample SD 4.86 points.

    - Prediction-flip rates were 10.53%, 5.26%, 2.63%, 21.62% and 13.51%.

    - Mean flip rate was 10.71% with sample SD 7.45%.

    - Predictions were therefore not invariant to the intervention, but its effect on accuracy had no consistent fold direction.

- PROTEINS showed stronger but highly variable fitted-model sensitivity.

    - Delta-loss folds were 0.1311, 0.0049, 0.0047, 0.1029 and 0.0059.

    - Delta loss was positive in all five folds.

    - Mean delta loss was 0.0499 with sample SD 0.0621.

    - Delta-accuracy folds were -17.94, -1.79, +1.79, -10.81 and 0.00 percentage points.

    - Mean delta accuracy was -5.75 percentage points with sample SD 8.36 points.

    - Prediction-flip rates were 34.08%, 4.48%, 2.69%, 22.52% and 2.70%.

    - Mean flip rate was 13.30% with sample SD 14.31%.

    - The classification effect therefore varied substantially between folds even though intervention loss worsened in every fold.

- NCI1 showed the clearest and most consistent fitted-model sensitivity.

    - Delta-loss folds were 0.3345, 0.6011, 0.3192, 0.5590 and 0.3025.

    - Every fold therefore had higher cross-entropy after the intervention.

    - Mean delta loss was 0.4233 with sample SD 0.1443.

    - Delta-accuracy folds were -18.49, -21.90, -19.46, -20.92 and -20.80 percentage points.

    - Every fold therefore had lower accuracy after learned attention scoring was removed.

    - Mean delta accuracy was -20.32 percentage points with sample SD only 1.34 points.

    - Prediction-flip rates were 37.47%, 44.53%, 41.12%, 47.69% and 38.32%.

    - Mean flip rate was 41.82% with sample SD 4.28%.

    - These results provide strong descriptive evidence that the fitted NCI1 GAT predictions depended materially on their learned attention scoring.

- RQ4 therefore produced a layer-dependent and dataset-dependent answer.

    - First-layer attention remained close to uniform on average across all three datasets.

    - Second-layer attention was moderately non-uniform on MUTAG, close to uniform on PROTEINS and substantially non-uniform on NCI1.

    - Removing learned attention scoring produced modest mixed accuracy effects on MUTAG, variable effects on PROTEINS and a large consistent degradation on NCI1.

    - Learned attention weighting was therefore neither universally essential nor universally dispensable under the fitted-model intervention.

- RQ3 and RQ4 gave importantly different evidence because retraining and fitted intervention answer different questions.

    - RQ3 allowed all remaining trainable parameters of the Uniform GAT to adapt during 500 training epochs.

    - RQ4 removed attention scoring only after an ordinary learned-attention GAT had already been fitted.

    - On NCI1, RQ3 found only a 1.22 percentage-point mean advantage for learned-attention retraining, while RQ4 reduced fitted-model accuracy by 20.32 percentage points on average.

    - On PROTEINS, the RQ3 learned-minus-uniform retraining difference was only 0.54 points, while the fitted intervention changed accuracy by -5.75 points on average.

    - On MUTAG, the retrained uniform model had the higher RQ3 mean by 1.59 points, while the fitted intervention produced a smaller and mixed -1.10-point mean change.

    - The results therefore show why a small retraining difference cannot be interpreted as evidence that an already fitted learned-attention model is insensitive to its attention mechanism.

- The observed concentration values were not interpreted as direct measures of predictive importance.

    - NCI1 combined the largest Conv2 departure with the largest fitted intervention effect.

    - PROTEINS showed that relatively small average attention departure can still coexist with substantial prediction changes in some folds.

    - The intervention changes attention scoring in both layers simultaneously.

    - The study therefore cannot attribute the prediction effects specifically to Conv1, Conv2 or to the magnitude of the entropy measure.

- The RQ4 evidence does not establish attention as a faithful explanation method.

    - Attention departure describes realised neighbourhood-weight concentration.

    - Intervention sensitivity describes how the fitted model's predictions respond when that learned weighting is removed.

    - Neither quantity proves that high-attention neighbours are causally important in the underlying molecular or protein domain.

    - No claim of statistical significance or equivalence is made from the five folds.

    - Sample SD describes variation among the five observed fold values and is not a confidence interval.

- The planned RQ4 scope is complete.

    - No head-similarity catalogue was added.

    - No self-loop-mass analysis was added.

    - No random perturbation control was added.

    - No additional fitted intervention was selected after observing these outcomes.

    - The saved JSON retains the graph-level evidence needed to reconstruct the reported fold and dataset summaries.

- Commit: 7.4 analysed fitted reference-GAT attention and uniform intervention





# 7.5 Cross-Experiment Evidence Notes

- Mapped the four Stage 7 research questions to their exact experimental evidence, observed answer and main qualification.

    - The four questions form one connected investigation of graph attention design rather than four independent experiments.

    - RQ1 changes the organisation of multi-head attention while fixing total first-layer width.

    - RQ2 changes the attention-scoring formulation from standard GAT to standard general-form GATv2.

    - RQ3 compares learned attention with a separately retrained uniform-attention GAT.

    - RQ4 characterises the coefficients learned by fitted reference GATs and measures the sensitivity of those same fitted models to removing learned attention scoring without retraining.

    - Negative, mixed and dataset-dependent outcomes were retained rather than converted into a single preferred attention configuration.

- RQ1 asked how first-layer GAT head count affects graph-classification performance when total first-layer representation width remains fixed at 64.

    - Heads 1, 2, 4 and 8 were compared on MUTAG, PROTEINS and NCI1.

    - Their channels per head were 64, 32, 16 and 8 respectively.

    - Total first-layer width and overall GAT parameter count therefore remained fixed within each dataset.

    - MUTAG mean accuracies were 76.51%, 77.07%, 75.48% and 79.23% for 1, 2, 4 and 8 heads.

    - PROTEINS means were 73.94%, 73.40%, 73.94% and 74.12%.

    - NCI1 means were 71.31%, 72.51%, 72.09% and 72.31%.

    - The observed relationship was non-monotonic on every dataset.

    - Eight heads had the highest observed mean on MUTAG and PROTEINS, while two heads had the highest observed mean on NCI1.

    - PROTEINS showed only a 0.72 percentage-point range between the highest and lowest observed means.

    - No head count was highest on every dataset or every outer fold.

- The observed answer to RQ1 is therefore that head count affected performance in a dataset-dependent way under fixed total width, with no evidence of a general monotonic benefit from increasing the number of heads.

    - The comparison isolates head organisation more cleanly than a design in which additional heads also increase total representation width.

    - It does not establish a universally optimal head count.

    - It also does not test the effect of increasing head count and total representation width together.

    - The fold means and sample standard deviations are descriptive summaries of five outer folds rather than significance or equivalence tests.

- A possible explanation for the non-monotonic RQ1 behaviour is the fixed-width trade-off between attention diversity and per-head channel width.

    - More heads provide more separately learned attention distributions.

    - Under fixed total width, however, each individual head becomes narrower.

    - Redundant head behaviour could also reduce the benefit of adding heads.

    - These are mechanism-based explanations consistent with the design, not effects separately identified by the experiment.

- RQ2 asked how standard GAT compares with standard general-form GATv2 under the common graph-classification pipeline.

    - The comparison reused the 30 existing reference fits and required no new optimisation.

    - MUTAG mean accuracy was 79.23% for GAT and 81.34% for GATv2.

    - The paired GATv2-minus-GAT fold differences on MUTAG were +7.89, 0.00, +2.63, -5.41 and +5.41 percentage points.

    - Their mean was +2.11 percentage points with sample SD 5.14 points.

    - PROTEINS mean accuracy was 74.12% for GAT and 73.22% for GATv2.

    - Its paired differences were -1.35, +0.45, -0.90, -1.35 and -1.35 points.

    - Their mean was -0.90 points with sample SD 0.78 points.

    - NCI1 mean accuracy was 72.31% for GAT and 71.97% for GATv2.

    - Its paired differences were +0.85, 0.00, -0.61, -1.70 and -0.24 points.

    - Their mean was -0.34 points with sample SD 0.93 points.

- The observed answer to RQ2 is therefore that GATv2 did not provide a consistent predictive advantage over GAT under this project pipeline.

    - GATv2 had the higher observed mean on MUTAG.

    - GAT had the higher observed mean on PROTEINS and NCI1.

    - The theoretical ability of GATv2 to use query-dependent neighbour ranking therefore did not imply an accuracy improvement on every dataset.

    - The result does not contradict the theoretical distinction between static GAT ranking and dynamic GATv2 ranking.

- RQ2 is not a capacity-equal causal test of dynamic attention ranking.

    - MUTAG GAT contained 5,058 parameters while GATv2 contained 9,730.

    - PROTEINS counts were 4,802 and 9,218.

    - NCI1 counts were 6,978 and 13,570.

    - The project intentionally retained the standard general-form GATv2 with share_weights=False rather than forcing its parameter count to match GAT.

    - Any observed accuracy difference can therefore reflect the complete model formulations rather than only the static-versus-dynamic ranking distinction.

- RQ3 asked whether learning non-uniform GAT neighbourhood coefficients improves performance relative to training the same GAT architecture with uniform attention throughout optimisation.

    - The learned condition reused the 15 reference GAT fits.

    - The uniform condition used 15 fresh fits with both layers' attention-scoring tensors fixed at zero.

    - MUTAG learned GAT mean accuracy was 79.23% and Uniform GAT mean accuracy was 80.83%.

    - Learned-minus-uniform fold differences were -2.63, -2.63, 0.00, -2.70 and 0.00 percentage points.

    - Their mean was -1.59 points with sample SD 1.45 points.

    - PROTEINS means were 74.12% learned and 73.58% uniform.

    - Its fold differences were +0.45, +0.45, -1.79, +3.60 and 0.00 points.

    - Their mean was +0.54 points with sample SD 1.95 points.

    - NCI1 means were 72.31% learned and 71.09% uniform.

    - Its fold differences were +0.36, +0.61, +1.95, +1.09 and +2.07 points.

    - Their mean was +1.22 points with sample SD 0.77 points.

- The observed answer to RQ3 is therefore dataset-dependent.

    - Uniform-attention retraining had the higher observed mean on MUTAG.

    - PROTEINS showed only a small mean difference with mixed fold directions.

    - NCI1 favoured learned attention in all five outer folds.

    - The NCI1 result is the clearest descriptive evidence in RQ3 for a performance benefit from learning non-uniform coefficients during training.

    - The results do not establish that learned attention is universally beneficial or universally unnecessary.

    - Small retraining differences do not establish statistical equivalence.

- RQ3 changes the optimisation problem as well as the attention coefficients.

    - A Uniform GAT is trained from the beginning under the uniform-attention constraint.

    - Its message transformations and classifier can therefore adapt throughout optimisation to that constraint.

    - RQ3 cannot establish whether an already fitted learned-attention GAT would remain unchanged if its learned weighting were removed after training.

    - That distinction motivated the fitted-model component of RQ4.

- RQ4 first asked how far the attention coefficients of the fitted reference GATs depart from uniform neighbourhood weighting.

    - Departure was measured as one minus normalised attention entropy.

    - Zero denotes uniform weighting and larger values denote more concentrated attention.

    - Conv1 departure was calculated separately for each of its eight heads before averaging heads at receiver level.

    - Eligible receivers were averaged within graphs and graphs equally within outer folds.

    - Conv1 mean departures were 0.0091 on MUTAG, 0.0092 on PROTEINS and 0.0268 on NCI1.

    - Their sample SDs were 0.0082, 0.0068 and 0.0108 respectively.

    - Conv1 therefore remained close to uniform on average across all three datasets under this aggregate measure.

- Conv2 attention departure was much more dataset-dependent.

    - MUTAG Conv2 fold departures were 0.0654, 0.1808, 0.2481, 0.0366 and 0.0518.

    - Its mean was 0.1165 with sample SD 0.0931.

    - PROTEINS fold departures were 0.0408, 0.0000, 0.0000, 0.0495 and 0.0000 at four-decimal reporting precision.

    - Its mean was 0.0181 with sample SD 0.0249.

    - NCI1 fold departures were 0.3198, 0.3015, 0.2017, 0.3774 and 0.2105.

    - Its mean was 0.2822 with sample SD 0.0750.

    - NCI1 therefore showed the strongest and most consistently non-uniform second-layer attention of the three datasets.

- RQ4 then asked how the same fitted reference GATs respond when their learned attention scoring is replaced by uniform weighting without retraining.

    - Only conv1.att_src, conv1.att_dst, conv2.att_src and conv2.att_dst were zeroed in the intervention copy.

    - Every other fitted state tensor was preserved.

    - MUTAG mean delta loss was +0.0346 with sample SD 0.0404.

    - MUTAG mean delta accuracy was -1.10 percentage points with sample SD 4.86 points.

    - Its mean prediction-flip rate was 10.71% with sample SD 7.45%.

    - The MUTAG accuracy effect had mixed fold directions.

    - PROTEINS mean delta loss was +0.0499 with sample SD 0.0621.

    - PROTEINS mean delta accuracy was -5.75 percentage points with sample SD 8.36 points.

    - Its mean prediction-flip rate was 13.30% with sample SD 14.31%.

    - PROTEINS therefore showed stronger but highly variable fitted-model sensitivity.

    - NCI1 mean delta loss was +0.4233 with sample SD 0.1443.

    - NCI1 mean delta accuracy was -20.32 percentage points with sample SD 1.34 points.

    - Its mean prediction-flip rate was 41.82% with sample SD 4.28%.

    - Every NCI1 fold had higher loss and lower accuracy after the intervention.

- The observed answer to RQ4 is therefore strongly layer-dependent and dataset-dependent.

    - First-layer attention remained close to uniform on average on all three datasets.

    - Second-layer attention ranged from near-uniform on PROTEINS to substantially non-uniform on NCI1.

    - Removing learned attention scoring produced modest mixed accuracy effects on MUTAG, highly variable effects on PROTEINS and a large consistent degradation on NCI1.

    - Learned attention weighting was therefore neither universally essential nor universally dispensable for the fitted reference GATs.

- RQ3 and RQ4 provide complementary rather than contradictory evidence.

    - On MUTAG, Uniform GAT retraining exceeded the learned GAT mean by 1.59 percentage points, while the fitted intervention changed accuracy by -1.10 points on average with mixed fold directions.

    - On PROTEINS, learned-attention retraining exceeded uniform retraining by only 0.54 points on average, while the fitted intervention changed accuracy by -5.75 points on average.

    - On NCI1, learned-attention retraining exceeded uniform retraining by 1.22 points on average, while the fitted intervention reduced accuracy by 20.32 points on average.

    - A separately retrained uniform model can adapt its remaining parameters to the uniform-attention constraint.

    - An already fitted learned-attention model subjected to the RQ4 intervention receives no opportunity for such compensation.

    - A small RQ3 retraining difference therefore cannot be interpreted as evidence that the fitted learned-attention solution is insensitive to its attention mechanism.

- Attention concentration and fitted-model sensitivity were not treated as equivalent quantities.

    - NCI1 combined the largest Conv2 departure with the largest and most consistent intervention effect.

    - PROTEINS nevertheless showed that small average attention departure can coexist with substantial prediction changes in some folds.

    - RQ4 intervened on attention scoring in both layers simultaneously.

    - The experiment therefore does not establish that Conv2 concentration caused the observed NCI1 intervention effect.

    - It also does not establish a general quantitative relationship between entropy departure and predictive importance.

- Several possible explanations remain distinct from the measured observations.

    - RQ1 behaviour may reflect a trade-off between the number of independently learned heads and the reduced channel width available to each head under fixed total representation width.

    - RQ2 behaviour may partly reflect the larger parameterisation of the chosen general-form GATv2 as well as its different scoring mechanism.

    - RQ3 and RQ4 differ because retraining permits optimisation of the remaining parameters under the imposed attention condition while the fitted intervention does not.

    - Dataset size, graph structure, feature distributions and finite-sample variation may contribute to the different fold behaviours observed across MUTAG, PROTEINS and NCI1.

    - These possibilities were not isolated experimentally and are therefore explanatory hypotheses rather than findings.

- All Stage 7 conclusions are conditional on the project's controlled feature and optimisation choices.

    - Principal models use categorical node features and graph connectivity.

    - Continuous node attributes and edge features are excluded from the core forwards.

    - The findings therefore describe attention behaviour under this feature-access policy rather than every possible representation of the three datasets.

    - Every final fit used the same two-layer architecture policy, hidden width 64, global sum readout, fixed optimisation settings, five-fold assessment procedure and predetermined seed schedule.

    - This consistency strengthens within-project comparisons but does not make the results universal beyond the studied configurations and datasets.

- Fold-level uncertainty was retained explicitly.

    - Every reported Stage 7 performance comparison preserves all five outer-fold values.

    - Arithmetic means summarise those five observations.

    - Sample standard deviations describe variation among them.

    - The folds have overlapping fitting sets and are not independent repeated-seed trials.

    - No significance, confidence-interval or equivalence claim is made from the five folds.

    - Mixed fold directions remain part of the evidence rather than being hidden by mean values.

- The combined Stage 7 evidence does not support a single universally preferable attention design.

    - More attention heads did not monotonically improve accuracy.

    - GATv2 did not consistently outperform standard GAT.

    - Learned non-uniform attention during retraining did not improve mean accuracy on every dataset.

    - Fitted-model sensitivity to removing learned weighting varied substantially between datasets.

    - The strongest consistent attention-specific effect was the NCI1 fitted intervention, but that result is not promoted into a universal claim.

- The Stage 7 evidence is sufficient for the planned core attention investigation.

    - No additional intervention, synthetic task, dataset or post hoc attention statistic was introduced to obtain a cleaner result.

    - No observed unfavourable or inconsistent outcome was used to redefine the research questions.

    - The substantive broader Discussion remains for the report phase.

    - The next Stage 7 work is Closing Notes followed by the complete Review 7.

- Commit: 7.5 synthesised the core attention evidence