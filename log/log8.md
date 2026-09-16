# 8.1 Core Result Set Closure

- Established the complete core result set from the supplied repository archive and the recorded Stage 6 and Stage 7 validation evidence.

    - The reference matrix contained 75 fits covering GCN, GraphSAGE, GIN, GAT and GATv2 on MUTAG, PROTEINS and NCI1 across five outer folds.

    - The additional-head matrix contained 45 GAT fits covering first-layer head counts of 1, 2 and 4 on the same datasets and folds, with total first-layer width fixed at 64.

    - The uniform-attention matrix contained 15 Uniform GAT fits covering the same datasets and folds.

    - This gave 135 distinct optimisation fits across 27 configuration and dataset groups.

    - Every expected JSON record had a non-empty paired PT file containing the saved selected-state output.

- Retained the exact filename conventions identifying the final result groups.

    - Reference records used results/cross_validation_{model}_{dataset}_fold{fold_id}_seed{seed}.json, with lower-case model and dataset names.

    - Additional-head records used results/cross_validation_gat_{dataset}_heads{heads}_fold{fold_id}_seed{seed}.json, with heads equal to 1, 2 or 4.

    - Uniform records used results/cross_validation_gat_{dataset}_uniform_fold{fold_id}_seed{seed}.json.

    - Each group contained outer-fold IDs 0 through 4, with training seed equal to 2000 + fold_id.

    - The paired selected-state path used the same basename with a .pt extension.

    - These identities kept the final cross-validation groups separate from historical development and epoch-budget diagnostic records.

- Preserved reference reuse across the research questions.

    - RQ1 combined the 45 additional-head fits with the 15 existing eight-head reference GAT fits.

    - RQ2 reused the 15 reference GAT and 15 reference GATv2 fits.

    - RQ3 compared the 15 reference GAT fits with the 15 Uniform GAT fits.

    - RQ4 reused the 15 selected reference GAT states without additional optimisation.

    - A reference fit used in more than one comparison remained one fit in the total of 135.

- Retained the successful Stage 7 execution of python -m experiments.summarise as the existing validation and summary evidence for the reference groups and RQ1 through RQ3.

    - experiments/result_validation.py checked the expected records against the loaded datasets, reconstructed partitions, effective settings, checkpoint-selection metadata, prediction-derived accuracies and recorded runtime conditions.

    - The summariser retained all five outer-fold values and calculated their arithmetic mean and sample standard deviation.

    - Parameter counts and training-pass times remained available alongside the accuracy summaries.

    - The archive inspection confirmed the expected record identities, paired-state availability, 500 completed epochs per fit and agreement between stored accuracies and predictions.

    - State-file presence established that the paired outputs were available. It did not independently establish their tensor contents or reproduce model predictions.

- Confirmed complete RQ4 coverage in results/rq4_fitted_gat_attention.json.

    - The saved analysis contained 15 selected reference GAT states and recorded zero new optimisation fits.

    - MUTAG contained five outer folds covering all 188 graphs once.

    - PROTEINS contained five outer folds covering all 1,113 graphs once.

    - NCI1 contained five outer folds covering all 4,110 graphs once.

    - The complete analysis therefore covered 5,411 distinct graphs, each assigned to its corresponding outer-test fold.

- Reconciled the saved RQ4 fold records with the corresponding reference GAT records.

    - Original graph IDs and their order matched the reference test_indices.

    - Labels and learned predictions matched the saved reference labels and predictions.

    - The recorded selected epochs and model-state paths identified the same selected reference states.

    - Learned accuracies matched the reference accuracies, and learned losses agreed within the established numerical tolerance.

    - Recorded device and software information agreed with the reference records.

- Reconciled the RQ4 summaries with their saved constituent values.

    - For each layer, averaging graph-level attention departures reproduced the recorded fold mean.

    - The five fold values reproduced the stored arithmetic mean and sample standard deviation.

    - Summing the graph-level receiver counts reproduced the recorded eligible-receiver and single-entry-receiver totals.

    - Comparing intervention predictions with labels reproduced intervention accuracy.

    - Comparing intervention predictions with learned predictions reproduced the prediction-flip rate.

    - Subtracting learned accuracy from intervention accuracy reproduced delta accuracy.

    - Subtracting learned loss from intervention loss reproduced delta loss.

    - These operations checked consistency of the saved evidence. They did not repeat attention extraction or recompute cross-entropy from model outputs.

- Preserved the scientific source commits recorded by the completed experiments.

    - Reference fits retained c60bbd57fd1c7cc7e6bce3a4a51c6fdc54b0776f.

    - Additional-head fits retained 59b4777610a56b06fb9dec3c3c2da8b9416791b0.

    - Uniform-attention retraining retained f68e626a88c54c160ea47b1d17d1b9c7e710ec7f.

    - The retained RQ4 analysis used f94c44c67f9d02bbaaf99f8f7bf01690f1c7de2c and identified the reference-fit source separately.

    - These source identities remained distinct from later results and documentation commits.

- Retained the methodological qualifications governing the final evidence.

    - Results described the fixed project pipeline on MUTAG, PROTEINS and NCI1 using categorical node features and graph connectivity.

    - The shared 500-epoch budget was informed by development validation on benchmark graphs later reused for cross-validation. It was fixed before CV outcomes but was not independent of all assessment data.

    - One fit per outer fold did not separate initialisation variability from partition effects.

    - Sample standard deviation described variation among the five observed fold values. The overlapping training sets did not provide five independent experimental datasets.

    - The GAT and GATv2 comparison matched the assessment pipeline while retaining different parameterisations.

    - Uniform retraining measured performance after adaptation during optimisation, whereas RQ4 measured sensitivity of the existing fitted model without adaptation.

    - Attention concentration and fitted-intervention sensitivity did not establish explanation faithfulness or real-world causal importance.

- The closure inspection identified no missing core record or inconsistency in the examined evidence.

    - The completed result groups and selected states were retained as the final evidence for the report phase.

    - No additional optimisation, model inference or change to the scientific source was required.

- Commit: 8.1 closed the core result set





# 8.2 Research Findings and Source Notes

- Consolidated the completed research questions into a compact evidence and interpretation map.

    - The result groups were those identified in 8.1.

    - Numerical findings were reconciled with the preserved records and the Stage 7 evidence notes.

    - The table distinguishes the observed answer from claims that the comparison cannot establish.

    | Research question | Preserved evidence | Supported answer | Main qualification |
    | --- | --- | --- | --- |
    | RQ1: First-layer head count at fixed total width | Reference GAT and heads1, heads2 and heads4 records on all three datasets | Accuracy did not increase monotonically with head count. Eight heads had the highest observed mean on MUTAG and PROTEINS; two heads had the highest on NCI1. | No universally optimal head count was established. Increasing heads also reduced channels per head under the fixed-width design. |
    | RQ2: Standard GAT versus general-form GATv2 | Matched reference GAT and GATv2 records | GATv2-minus-GAT mean accuracy differences were +2.11, -0.90 and -0.34 percentage points on MUTAG, PROTEINS and NCI1 respectively. | The chosen GATv2 had more parameters. The comparison did not isolate the causal effect of query-dependent ranking. |
    | RQ3: Learned versus uniform-attention retraining | Reference GAT and uniform records | Learned-minus-uniform mean accuracy differences were -1.59, +0.54 and +1.22 percentage points on MUTAG, PROTEINS and NCI1 respectively. | Both conditions could adapt during training. Small differences did not establish equivalence or fitted-model insensitivity. |
    | RQ4: Fitted attention and uniform intervention | results/rq4_fitted_gat_attention.json and its 15 selected reference GAT states | Conv1 was close to uniform on average; Conv2 was more dataset-dependent. Intervention-minus-learned mean accuracy differences were -1.10, -5.75 and -20.32 percentage points on MUTAG, PROTEINS and NCI1 respectively. | Both layers were intervened on together. Concentration and prediction sensitivity did not establish explanation faithfulness or domain causality. |

- Retained fold variability alongside the mean paired differences.

    - RQ2 difference sample standard deviations were 5.14, 0.78 and 0.93 percentage points for MUTAG, PROTEINS and NCI1 respectively.

    - RQ3 difference sample standard deviations were 1.45, 1.95 and 0.77 percentage points.

    - RQ4 accuracy-difference sample standard deviations were 4.86, 8.36 and 1.34 percentage points.

    - Every NCI1 fold favoured learned attention over uniform retraining in RQ3.

    - Every NCI1 fold had higher loss and lower accuracy after the fitted intervention in RQ4.

    - Means and sample standard deviations remained descriptive summaries of the five outer folds.

- Preserved both components of RQ4 in the report evidence.

    - Conv1 mean departures from uniform weighting were 0.0091 on MUTAG, 0.0092 on PROTEINS and 0.0268 on NCI1.

    - Conv2 mean departures were 0.1165, 0.0181 and 0.2822 respectively.

    - These were means of the five graph-equal fold summaries, with the layers kept separate.

    - Mean intervention-minus-learned cross-entropy differences were +0.0346, +0.0499 and +0.4233 respectively.

    - Mean prediction-flip rates were 10.71%, 13.30% and 41.82% respectively.

    - Complete fold values, sample standard deviations and receiver counts remained in the saved RQ4 record.

    - Rounded zero departures were not treated as proof of exact uniformity.

- The relationship between RQ3 and RQ4 remained central to the interpretation.

    - On NCI1, learned attention exceeded uniform retraining by 1.22 percentage points on average.

    - Removing learned scoring from the already fitted reference GAT reduced accuracy by 20.32 percentage points on average.

    - Uniform retraining allowed the remaining transformations and classifier to adapt throughout optimisation.

    - The fitted intervention preserved those parameters and provided no opportunity to adapt.

    - The two results therefore described different experimental conditions rather than contradictory estimates of the same effect.

- Revisited Graph Attention Networks by Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò and Yoshua Bengio.

    - The paper was published at ICLR 2018. The supplied copy was arXiv:1710.10903v3, dated 4 February 2018.

    - Section 2.1, particularly Equations 2 through 6 on pages 3 and 4, defined neighbourhood softmax, weighted aggregation and multi-head concatenation or averaging.

    - Section 3.3 on pages 6 and 7 supplied the eight-head, eight-channel precedent and described the constant-attention control.

    - The constant-attention comparison concerned node-level multi-label prediction on PPI. Processing multiple graphs did not make its target graph classification.

    - RQ3 therefore used an established control idea in the project's graph-classification setting rather than introducing constant attention as a new method.

    - The paper's task-specific activations, regularisation, output layers and stopping procedure were not the complete recipe used by this project.

- Revisited How Attentive are Graph Attention Networks? by Shaked Brody, Uri Alon and Eran Yahav.

    - The paper was published at ICLR 2022. The supplied copy was arXiv:2105.14491v3, dated 31 January 2022.

    - Sections 3.1 through 3.3 on pages 4 and 5 defined static and dynamic attention and established the GAT ranking restriction and GATv2 reformulation.

    - Static ranking applied separately within each head for fixed input representations. It did not imply identical normalised coefficients for every receiver.

    - Section 4's Setup paragraph on page 6 described shared transformations in the main GATv2 experiments.

    - Appendix G.2 and Table 18 on page 26 distinguished the general and experimental parameterisations. Those calculations concerned one layer and one head and omitted biases.

    - The project retained share_weights=False and reported actual complete-model parameter counts.

    - The expressivity result motivated RQ2 but did not guarantee improved graph-classification accuracy under the chosen finite architecture and training procedure.

- Revisited A Fair Comparison of Graph Neural Networks for Graph Classification by Federico Errica, Marco Podda, Davide Bacciu and Alessio Micheli.

    - The paper was published at ICLR 2020. The supplied copy was arXiv:1912.09893v3, dated 17 February 2022.

    - The revised preprint date did not change the conference publication year.

    - Section 3 on page 3 distinguished model selection from model assessment.

    - Section 5's Experimental Setting on page 6 used ten outer folds, an inner 90/10 holdout, configuration search and three retrainings after selection, with validation-based early stopping.

    - Precomputed stratified partitions supported consistent comparisons across models.

    - The project adopted separation of fold-local checkpoint selection and assessment, together with shared partitions, while retaining its own five-fold, fixed-configuration, one-fit-per-fold procedure.

    - The earlier development-validation choice of the common epoch budget remained a separate benchmark-reuse limitation.

- Kept the evidence for RQ4's measurement procedure distinct from the literature motivation.

    - Normalised-entropy departure, receiver eligibility, head averaging, graph-equal aggregation and the simultaneous two-layer intervention were the project's predeclared analysis choices.

    - Their specification was retained in log/log6.md, their implementation in experiments/analyse_fitted_gat.py and their execution and interpretation in log/log7.md.

    - The inspected papers were not presented as having performed this exact analysis or established the resulting project findings.

- Identified the source files needed to explain and reproduce the implemented study.

    - experiments/train_cv.py owned the shared assessment settings, reference model settings, dataset list, fold schedule and cross-validation runner.

    - experiments/ablation.py defined the additional head variants, Uniform GAT and reference reuse.

    - experiments/runners/train_heads.py and experiments/runners/train_uniform.py provided the explicit additional-fit entry points.

    - src/data.py contained dataset loading, seeding and partition construction.

    - src/models contained the five classifier definitions and the shared model factory.

    - src/training.py implemented fitting and validation-selected state restoration.

    - src/evaluation.py produced graph-mean loss, accuracy and aligned predictions and labels.

    - src/recording.py preserved result paths, selected states, runtime and source information.

    - experiments/result_validation.py and experiments/summarise.py supported validation and presentation of the reference and RQ1 through RQ3 evidence.

    - experiments/analyse_fitted_gat.py defined the completed RQ4 computation.

- Retained the selected states and development evidence needed for the methodological account.

    - The 135 paired PT files preserved the selected states for the final optimisation fits.

    - The 15 reference GAT states were the fitted inputs to RQ4.

    - RQ4 preserved its measurements in one JSON file and did not create another set of trained checkpoints.

    - results/development_epoch_budget_audit.json and experiments/inspections/inspect_epoch_budget.py retained the evidence and procedure behind the common 500-epoch decision.

    - The audit record identified source commit 834ca47202de75bd64cd89dd4bec17f36d2ccb70.

    - Final fit and analysis source commits remained those recorded in 8.1 and in the corresponding result files.

    - requirements.txt and the runtime fields in the records retained the package and execution context.

- Identified potential report assets and their evidence sources.

    - A dataset table could use the recorded inspections in log/log1.md and log/log6.md, supported by the current dataset inspectors and loader policy.

    - An architecture and settings table could use the model definitions, effective result settings and recorded methodological provenance.

    - A reference-results table could present all five models and also provide the main RQ2 comparison.

    - A head-count table could present all four configurations at fixed total width. A head-count figure remained optional if it communicated the pattern more clearly.

    - A uniform-retraining table could present the learned and uniform results with their paired differences and trainable parameter counts.

    - An RQ4 table could present both layers' departures together with loss changes, accuracy changes and prediction-flip rates.

    - Complete fold values could support a purposeful appendix without repeating every value in the main text.

    - Numerical assets would be generated from the preserved records. RQ4 tables would read the saved analysis JSON without repeating the fitted-model analysis.

- Inspected the current report skeleton and bibliography.

    - report/main.tex retained the empty agreed section structure.

    - report/references.bib contained the existing kipf2017gcn and velickovic2018gat entries.

    - The verified GAT author order and conference year agreed with the existing entry.

    - GATv2 and Fair Comparison metadata were retained here for bibliography preparation during the report phase.

    - No measured report figures were present in the supplied archive. The assets listed above remained proposed outputs rather than completed figures or tables.

- Consolidated the report evidence boundary.

    - Project records supported numerical and implementation claims.

    - Inspected paper passages supported the theoretical motivation and comparisons with prior methods and evaluation practice.

    - Possible explanations involving head diversity, optimisation, parameterisation and dataset structure remained hypotheses where the experiments had not isolated them.

    - The completed findings supported qualified answers to all four questions while preserving negative, mixed and dataset-dependent results.

- Commit: 8.2 consolidated research findings and source notes





# 8.3 Optional Traditional Baseline Disposition

- Deferred the optional graph-kernel and simple-classifier comparison.

    - No kernel method, classifier configuration or validation-based selection procedure had been established for this comparison.

    - Completing it would require a separate implementation and evaluation path.

    - The existing implementation used NumPy, PyTorch and PyTorch Geometric, with Matplotlib available for figures.

    - A graph-kernel and conventional classifier package was not part of the declared dependencies.

    - A sound traditional comparison remained a possible extension rather than introducing an inadequately specified baseline into the completed investigation.

- The deferral reflected implementation scope rather than an observed performance result.

    - No graph-kernel classifier was trained or assessed.

    - No conclusion was drawn about whether such a method would perform better or worse than the evaluated neural models.

    - The completed evidence remained 135 neural optimisation fits and the fitted-attention analysis of 15 selected reference GAT states.

- Retained the distinction between contextual neural baselines and traditional graph methods.

    - GCN, GraphSAGE and GIN provided contextual neural comparisons for GAT and GATv2.

    - These models did not substitute for an empirical graph-kernel comparison.

    - The results supported comparisons among the evaluated neural formulations and answers to the four attention questions under the fixed experimental conditions.

    - They did not establish superiority over graph kernels or non-neural graph classifiers.

- Identified the requirements for a possible future traditional comparison.

    - Specify the graph kernel, classifier and any selectable hyperparameters before assessment.

    - Use compatible graph information and document any differences in feature access.

    - Separate validation-based selection from outer-test assessment.

    - Preserve the selected settings, partitions, predictions and evaluation results.

    - Treat the existing neural findings as prior information rather than claiming that a subsequently designed comparison was specified independently of them.

- Commit: 8.3 deferred the optional traditional baseline





# 8.4 Research Evidence Consolidation

- Established the completed experimental scope.

    - The reference comparison covered GCN, GraphSAGE, GIN, GAT and GATv2 on MUTAG, PROTEINS and NCI1.

    - Each configuration used five outer folds with one fresh fit per fold.

    - The attention investigation covered fixed-width head count, standard GAT versus general-form GATv2, uniform-attention retraining and fitted-attention characterisation with a uniform intervention.

    - The complete evidence comprised 135 optimisation fits and the analysis of 15 selected reference GAT states.

    - The optional traditional graph-kernel comparison remained unperformed.

- Retained a traceable relationship between experimental settings, fitted states, predictions and summaries.

    - Each final fit's JSON record identified its effective settings, partitions, training seed, selected epoch, predictions, labels and scientific source commit.

    - The paired PT file preserved the selected model state.

    - experiments/result_validation.py and experiments/summarise.py supplied the existing validation and summary path for the reference results and RQ1 through RQ3.

    - results/rq4_fitted_gat_attention.json retained graph-level measurements, fold-level results and dataset summaries linked to the selected reference GAT states.

    - Final numerical presentation could therefore be generated from the preserved measurements without fitting additional models.

- Preserved the relationship between the research questions and their supported conclusions.

    - The fixed-width head study showed dataset-dependent, non-monotonic accuracy patterns.

    - General-form GATv2 did not consistently outperform GAT across the three datasets.

    - Learning attention during training did not provide a uniform benefit over the separately retrained uniform control.

    - Fitted-model sensitivity to removing learned scoring varied substantially across datasets, with the largest and most consistent accuracy reduction on NCI1.

    - The retraining and fitted-intervention results remained distinct because only retraining allowed the remaining parameters to adapt to uniform weighting.

- Retained the limits of those conclusions.

    - The findings concerned the chosen categorical node inputs, connectivity, two-layer model adaptations and fixed training procedure.

    - The earlier development-validation choice of the epoch budget limited claims of assessment independence on the reused benchmarks.

    - Five-fold sample standard deviation described observed variation rather than confidence, statistical significance or equivalence.

    - GATv2's different parameterisation prevented interpreting RQ2 as a pure causal estimate of dynamic ranking.

    - The simultaneous intervention on both GAT layers did not isolate the contribution of either layer.

    - Attention departure and prediction sensitivity did not establish explanation faithfulness or real-world causal importance.

    - The neural comparisons did not establish superiority over untested traditional graph methods.

- Preserved the distinction between measurements, theoretical results and possible explanations.

    - Numerical findings were supported by the saved experimental records.

    - The attention mechanisms and static-ranking distinction were supported by their mathematical formulations and the inspected primary literature.

    - The actual architecture, feature policy, selection procedure and intervention were established by the implemented source and recorded settings.

    - Explanations involving head diversity, optimisation, parameterisation or dataset structure remained hypotheses where their effects had not been separately identified.

- Commit: 8.4 consolidated the completed research evidence





# Stage 8 Closing Notes

- Decisions:

    - The completed neural investigation contains five reference models and four attention research questions on MUTAG, PROTEINS and NCI1.

    - The final optimisation evidence consists of 75 reference fits, 45 additional-head fits and 15 Uniform GAT fits, giving 135 distinct fits.

    - Reference reuse remains explicit. The eight-head GAT condition, GAT/GATv2 comparison and learned-attention side of the uniform comparison reuse the corresponding reference records.

    - RQ4 retains the analysis of 15 selected reference GAT states and introduces no additional optimisation fits.

    - The final record groups, selected states and scientific source identities are preserved. No missing core record or inconsistency was identified during the evidence reconciliation.

    - The optional traditional graph-kernel comparison is deferred. No result or claim of superiority over traditional graph methods is included.

    - Numerical presentation will use the preserved unrounded records. Display rounding will not change the values used to calculate means, differences or sample standard deviations.

    - Arithmetic mean accuracy and sample standard deviation across five outer folds remain the principal summaries. Paired accuracy differences are expressed in percentage points.

    - Dataset-specific findings remain separate. The results are not combined into a single ranking across heterogeneous datasets.

    - Uniform retraining and fitted intervention remain distinct experimental conditions. Adaptation during optimisation is available in the former and absent in the latter.

    - Attention concentration, fitted-model sensitivity, explanation faithfulness and real-world causal importance remain separate concepts.

    - Model comparisons retain their actual parameterisation differences. In particular, general-form GATv2 with share_weights=False is not treated as a parameter-matched causal control for dynamic ranking.

    - The common 500-epoch budget retains its development-validation-informed provenance on benchmark graphs later reused for CV.

    - The complete findings remain conditional on the selected datasets, categorical node inputs, connectivity, model adaptations, optimisation procedure and prescribed fold and seed schedule.

- Ideas:

    - A graph-kernel and simple-classifier comparison remains a possible future contextual extension, subject to a sound implementation and prospectively specified selection and assessment procedure.

    - No additional experimental direction was adopted during evidence consolidation.

- Report notes:

    - The 135 final JSON records preserve effective settings, partition indices, training seeds, selection metadata, aligned predictions and labels, parameter counts, runtime information and source commits.

    - Their paired PT files preserve the selected model states. Availability of a state file is distinct from independently reproducing its predictions.

    - experiments/result_validation.py and experiments/summarise.py provide the existing validation and summary path for the reference results and RQ1 through RQ3.

    - results/rq4_fitted_gat_attention.json preserves RQ4 graph-level measurements, fold-level outcomes and dataset summaries. Its numerical presentation can be regenerated from those saved values without repeating attention extraction or model inference.

    - RQ4 covers all 188 MUTAG graphs, 1,113 PROTEINS graphs and 4,110 NCI1 graphs once in their corresponding outer-test folds.

    - Reference fits retain source commit c60bbd57fd1c7cc7e6bce3a4a51c6fdc54b0776f.

    - Additional-head fits retain source commit 59b4777610a56b06fb9dec3c3c2da8b9416791b0.

    - Uniform-attention retraining retains source commit f68e626a88c54c160ea47b1d17d1b9c7e710ec7f.

    - The retained RQ4 analysis identifies source commit f94c44c67f9d02bbaaf99f8f7bf01690f1c7de2c and records the reference-state source separately.

    - RQ1 showed non-monotonic head-count results. Eight heads had the highest observed mean on MUTAG and PROTEINS, while two heads had the highest on NCI1. These observations did not establish a universal optimum.

    - RQ2 GATv2-minus-GAT mean accuracy differences were +2.11, -0.90 and -0.34 percentage points on MUTAG, PROTEINS and NCI1 respectively. Their sample standard deviations were 5.14, 0.78 and 0.93 points.

    - RQ3 learned-minus-uniform mean accuracy differences were -1.59, +0.54 and +1.22 percentage points on MUTAG, PROTEINS and NCI1 respectively. Their sample standard deviations were 1.45, 1.95 and 0.77 points.

    - RQ4 Conv1 mean departures from uniform weighting were 0.0091, 0.0092 and 0.0268 on MUTAG, PROTEINS and NCI1 respectively. Conv2 means were 0.1165, 0.0181 and 0.2822.

    - RQ4 intervention-minus-learned mean accuracy differences were -1.10, -5.75 and -20.32 percentage points on MUTAG, PROTEINS and NCI1 respectively. Their sample standard deviations were 4.86, 8.36 and 1.34 points.

    - The NCI1 retraining difference of +1.22 percentage points and fitted-intervention reduction of 20.32 points illustrate the distinction between learning under a constraint and imposing that constraint on an existing fitted solution.

    - RQ4 changed attention scoring in both layers simultaneously. The larger NCI1 Conv2 departure did not establish that Conv2 alone caused the intervention effect.

    - Graph Attention Networks, published at ICLR 2018, supplies the attention and multi-head formulations in Section 2.1 and the constant-attention control in Section 3.3. That control evaluated node-level predictions on PPI rather than graph-classification targets.

    - How Attentive are Graph Attention Networks?, published at ICLR 2022, supplies the static/dynamic distinction in Sections 3.1 through 3.3. Section 4 and Appendix G.2 distinguish its shared-transform experimental setup from the general parameterisation retained here.

    - A Fair Comparison of Graph Neural Networks for Graph Classification, published at ICLR 2020, motivates separating model selection from assessment. Its ten-fold configuration search and three retrainings differ from this project's fixed-configuration, five-fold, one-fit-per-fold procedure.

    - The final CV estimates are not an untouched external assessment independent of the earlier epoch-budget decision. results/development_epoch_budget_audit.json preserves the development evidence behind that choice.

    - Fold standard deviations include variation under different partitions and their prescribed training realisations. They do not separately estimate initialisation variability, and overlapping fitting sets limit independent-sample interpretations.

    - Potential report assets include dataset and architecture tables, the complete reference comparison, the fixed-width head comparison, the uniform-retraining comparison and the two-component RQ4 summary.

    - RQ2 can use the existing reference comparison rather than duplicating the same results in another full table. Additional figures are useful only where they clarify a pattern beyond the tables.

    - The supported overall conclusion is that attention-design effects depend on the dataset and experimental condition. The evidence does not establish a universally preferable head count, a consistent GATv2 advantage or a universal benefit or dispensability of learned attention.

- Code and evidence audit:

    - Audited all 33 active Python files against the implemented methodology and preserved experimental evidence. Archived implementations were excluded.

    - Source inspection confirmed fresh models and optimisers per fold, validation-only checkpoint selection, complete 500-epoch fitting, selected-state cloning and restoration, graph-weighted metrics and the declared training-time convention. No fold-level leakage or selection error was identified.

    - Confirmed the model dimensions, operator settings and attention controls. The head-count study retained fixed total width, GATv2 retained separate transformations, uniform retraining froze the four zeroed scoring tensors before optimisation, and the fitted intervention changed only those tensors without retraining.

    - Checked all 135 final JSON records. Effective settings, selection metadata, prediction-derived accuracies, timing arithmetic and recorded runtime conditions reconciled. Source identities agreed with the reference, additional-head and uniform-attention result groups.

    - Regenerated the partitions using the preserved graph-label ordering. Fitting, validation and outer-test indices were disjoint and complete, corresponding configurations used matching partitions, and every graph appeared in outer test exactly once.

    - Inspected all 135 paired selected-state files. Tensor names, shapes and parameter totals matched the model definitions, stored tensors were finite, and fixed GIN epsilon buffers remained zero. All 15 uniform-control states retained zero values in the four attention-scoring tensors.

    - Reconciled the RQ4 records for all 5,411 graphs with their reference folds, selected states, labels and learned predictions. Intervention accuracies, prediction-flip rates, recorded loss differences, graph-to-fold averages, sample standard deviations and receiver counts agreed with their constituent values.

    - Reconciled all 27 saved development epoch-budget trajectories. Earliest validation-loss minima, top-ranked epochs, cutoff gaps and the clean aggregate budget findings agreed with the recorded histories.

    - The audit checked source code and preserved evidence without fresh training or model inference. Partition reconstruction used recorded labels rather than an independent dataset reload, and checkpoint inspection did not independently reproduce predictions.

    - No scientific defect or evidence inconsistency requiring a research-code change or experimental rerun was identified. The validated implementation and numerical evidence were retained unchanged.

- Commit: Stage 8 recorded closing decisions and report notes