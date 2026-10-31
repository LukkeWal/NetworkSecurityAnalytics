# Network Security Analytics ML Pipeline

## Setup
Place the HIKARI-2021 data at `data/ALLFLOWMETER_HIKARI2021.csv`.

Run commands from the project directory. Install dependencies with 

`python -m pip install -r requirements.txt`.

Create the reproducible, label-stratified train/val/dev/test splits (60/10/20/10%):

`python main.py --init`

## Train a model
`main.py` trains binary classifiers by default; use `--multiclass` to predict attack categories.
Without filters, it runs all models, feature representations and resampling strategies. View available settings with:

`python main.py --help`

### Results
Decision thresholds are selected on `val` to maximize attack precision at recall of at least 95%, then models are evaluated
on `dev`. Use `--final-evaluation` to evaluate on the reserved `test` split.

Each model run appends a row with a unique ID to `results/binary_results.csv` (`results/classification_results.csv` for
`--multiclass`). Rows include model settings, training sample/class counts, accuracy, precision, recall, F1, false positive
rate and confusion matrices at default and selected thresholds, plus validation threshold metrics. Binary runs also record
ROC/precision-recall curves and AUCs; XGBoost and random forests record training/validation loss curves.

Add `--export` to save each evaluated model in a unique directory under `results/models/`, containing:

- `model.joblib`: fitted classifier and label encoder.
- `preprocessing.joblib`: fitted imputation values, scaling and PCA/UMAP transformations as applicable.
- `metadata.json`: feature representation and column order, classification mode, filtering settings, tuned threshold,
  original results CSV path and result ID.

Export runs compute fresh features to retain fitted transformations; they do not read or modify the feature cache.

### Train our best model
To train and evaluate our best performing model on the held out `test` data, run this PowerShell command:

`python --% main.py --model xgboost --final-evaluation --features raw_numeric --resampling no_resampling --early-stopping-rounds 20 --no-background train --model-params "{\"xgboost\":{\"max_depth\":6,\"min_child_weight\":5,\"learning_rate\":0.3,\"subsample\":1.0,\"colsample_bytree\":1.0}}"`

## Evaluate a saved model

`python main.py --evaluate-model results/models/<bundle>`

Evaluates on `data/splits/dev.csv`; add `--final-evaluation` to use `test.csv`. The path can also point to the bundle's
`metadata.json`. Keep all three bundle files together. Saved preprocessing, filtering and thresholds are reused without
fitting or tuning. Metrics are printed only; results CSVs and the feature cache are untouched. Training options cannot be
combined with `--evaluate-model`.

## Visualizing results
`Visualization.py` plots saved results using `--plot`, `--results <csv>` and `--id <result-id>`, or explores datasets with
`--plot` and `--data <csv>`. Figures are saved to `results/figures/` by default. View available plots and settings with:

`python Visualization.py --help`

## Python files

- `main.py`: Parses CLI options and coordinates split initialization, feature loading, training and evaluation.
- `DataHandler.py`: Reads traffic data, creates reproducible splits and prepares binary or multiclass labels.
- `PreProcessing.py`: Cleans numeric features, resamples training classes and builds raw, PCA and UMAP representations.
- `FeatureCache.py`: Saves and reuses prepared feature splits in `cache/features/` for each experiment configuration.
- `ModelArtifacts.py`: Exports model bundles with fitted preprocessing and evaluates saved models (cannot use cache).
- `Trainer.py`: Defines classifiers, trains parameter combinations and records supported training/validation loss curves.
- `Evaluate.py`: Selects validation thresholds, computes evaluation metrics and appends results to CSV files.
- `Visualization.py`: Generates plots from saved experiment results or dataset CSV files.
