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