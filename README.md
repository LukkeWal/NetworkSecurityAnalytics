Insert the data under:
`data/ALLFLOWMETER_HIKARI2021.csv`

Run from the project directory with the virtual environment activated:

```powershell
python main.py --help
```

With no flags, the pipeline runs all models, feature representations and resampling
strategies, using a total sample size of 10000 across train/validation/test.
Exploration reports and plots are disabled unless `--exploration` or
`--learning-curves` is supplied.

Without `--model`/`--models`, visualization flags run only the requested
visualizations and do not train or evaluate models. For example, plot the latest
recorded XGBoost and random forest runs without loading the dataset:

```powershell
python main.py --learning-curves
```

Use `--exploration` alone for dataset exploration, or combine both visualization
flags. To train and then plot, supply models explicitly, for example
`python main.py --model xgboost --feature raw_numeric --resampling no_resampling --learning-curves`.

Run just XGBoost on raw numeric features without resampling or exploration plots:

```powershell
python main.py --model xgboost --feature raw_numeric --sample-size 10000 --resampling no_resampling
```

Run both ensembles with custom settings (PowerShell):

```powershell
$modelParams = '{"xgboost":{"n_estimators":300,"max_depth":4,"learning_rate":0.05},"random_forest":{"n_estimators":300,"max_depth":12,"min_samples_leaf":5,"n_jobs":-1}}'
python main.py --models xgboost random_forest --features raw_numeric pca_95 --resampling no_resampling --model-params $modelParams
```

On Windows PowerShell 5.1, if JSON quotes are stripped when passing arguments to
Python, use `--model-params ($modelParams.Replace('"', '\"'))` instead.

| Flag | Purpose |
| --- | --- |
| `--models` / `--model` | One or more classifier names; default all available models |
| `--features` / `--feature` | One or more feature representations; default all |
| `--sample-size` / `--exploration-sample-size` | Positive total sample size; default 10000; capped at available data |
| `--resampling` | One or more of `no_resampling`, `undersample_majority`, `oversample_minority`; default all |
| `--model-params` | JSON object mapping selected model names to estimator parameters |
| `--early-stopping-rounds` | XGBoost validation log-loss patience; default 20; use 0 to disable |
| `--data` | Input CSV path; default `data/ALLFLOWMETER_HIKARI2021.csv` |
| `--results` | CSV to append results to; default `evaluation_results.csv` |
| `--figures` | Exploration and learning curve output directory; default `figures` |
| `--exploration` | Enable exploration reports and plots, including t-SNE; default disabled |
| `--learning-curves` | Enable learning curve plots; default disabled; losses are always recorded |

Each selected model runs for every selected representation and resampling strategy.
Unselected transformations are skipped. Sample size is applied before resampling.
Model hyperparameters are saved in the CSV's `settings.model_params` field.
The weighted XGBoost variant `xgboost_scale_pos_weight` requires binary labels.
For pipeline models (`svm`, `svm_balanced`, `knn`, `mlp`), use parameter names
such as `classifier__C` or `classifier__hidden_layer_sizes`.

Run CLI checks with `python -m unittest test_cli`.

XGBoost (including its weighted variant) uses early stopping by default: fitting
stops after 20 consecutive rounds without a new best validation log loss, or at
the `n_estimators` limit (100 by default). Threshold tuning and test predictions
automatically use the best iteration. The test split is not used for stopping.
Per-model `early_stopping_rounds` in `--model-params` overrides the CLI patience;
use JSON `null` to disable it for an individual model.

The CSV settings record `best_boosting_round` (starting at 1),
`best_validation_log_loss`, and `boosting_rounds_trained` when stopping is enabled.
The training curve includes all rounds actually fitted, including the patience
rounds after the best round. Random forest training is unaffected.

```powershell
python main.py --model xgboost --feature raw_numeric --resampling no_resampling --sample-size 100000 --early-stopping-rounds 20 --learning-curves
```
