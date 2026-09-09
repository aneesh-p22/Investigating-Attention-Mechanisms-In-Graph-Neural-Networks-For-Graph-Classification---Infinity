# 3.1 Shared Training and Evaluation Functions

- Extracted the existing training and evaluation functions so GraphSAGE and GIN can reuse the GCN training procedure

    - Each model accepts node features, connectivity and graph membership through model(x, edge_index, batch), and returns graph-class logits

    - The training code therefore works with the supplied model without needing to know which graph convolution it uses

- Moved train_epoch and train_model into src/training.py

    - train_epoch performs one pass through the training loader and returns loss and accuracy

    - train_model calls train_epoch and evaluate each epoch, selects the lowest-validation-loss state and restores it after fitting

    - These functions receive their inputs through arguments rather than reading the development runner's settings directly

- Moved evaluate into src/evaluation.py

    - Both validation and development-test assessment need the same loss and accuracy calculation without parameter updates

    - src/training.py imports evaluate for validation, while experiments/train.py imports it for development-test assessment

- Updated experiments/train.py to import the extracted functions and removed their original definitions

    - The runner still constructs the experiment: settings, data loaders, model and optimiser, followed by fitting, assessment and result recording

    - Passing model into train_model gives the function access to the same model object used by the runner

    - Restoring the selected parameters inside train_model therefore updates the model subsequently assessed by the runner

    - Function bodies, arguments and returned values were preserved, so the extraction changes code organisation without changing the intended training procedure

- Chose python -m src.training as the module-loading check

    - This loads src.training and its src.evaluation import

    - Function definitions are loaded without calling them, so no training or printed results are expected

    - Successful module loading checks the imports, but does not establish that a training run produces equivalent results

- Commit: 3.1 extracted shared training and evaluation functions