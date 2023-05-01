# Early-Prediction-of-Sepsis

## Dependencies

Execute `pip3 install -r requirements.txt` to install all the required libraries.

## Data

A minimal set of dataset is available at `data/`. They are copied from https://github.com/yinchangchang/DII-Challenge/tree/master/file.

## Code Execution
`python3 main.py` will perform the following:
- Pre-process raw data and save to `generated_data/` with `preprocessing/Preprocessor.py`.
- Read the processed data and set up custom `DataSet` and data loaders for training and validation.
- Initialize LSTM-based models, loss function, and optimizer.
- Use `tools/train_eval.py` to train the model.
- Save the trained model performance to `generated_models`.

This command accepts a variety of arguments, documented in `tools/parse.py`. The ones most relevant to the reproducibility project:
- `--epochs`: This controls the number of epochs for the training process. 20 is used in the project.
- `--use-glp`: This tells the command if the global pooling layer should be used (0 or 1). By default, the layer is enabled.
- `--use-cat`: This tells the command if the time encoding should be used (0 or 1). By default, the time encoding is enabled.
- `--use-avg-pooling`: This tells the command whether to use average pooling (0 or 1) if the global pooling is enabled. By default, max pooling would be used.

## Analysis
After `main.py` is ran, the following analysis can be done:
- `python3 analysis/lstm_ablation_comparison.py` will produce the charts to compare performance among different ablations.
- `python3 other_algorithms/other_testing.py` will product the charts to compare performance among LSTM-based model and other common ML models.

## Results
- `other_algorithms/`
    - `model_accuracies.png` demonstrates the accuracies from different models after being trained on the given dataset.
    - `roc_curves.png` demonstrates the AUROC values from different models after being trained on the given dataset.
- `analysis/`
    - `ablation_accuracies.png` demonstrates the accuracies from different ablations after being trained on the given dataset.
    - `ablation_aurocs.png` demonstrates the AUROC values from different ablations after being trained on the given dataset.

## Citations

Original paper:
- https://www.sciencedirect.com/science/article/pii/S266638992030266X

Reference code repo:
- https://github.com/yinchangchang/DII-Challenge
- https://github.com/Ichiro-Tai/Sepsis-Early-Detection

