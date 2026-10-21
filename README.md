# Network Security Analytics CLI

Run commands from the project directory. Install dependencies with `python -m pip install -r requirements.txt`.

```powershell
# Create reproducible train/val/development/test CSVs (60/10/20/10).
python main.py --init

# Train and evaluate a selected configuration.
python main.py --model xgboost --feature raw_numeric --resampling no_resampling

# Quick experiment using 10,000 rows from the training pool.
python main.py --model xgboost --feature raw_numeric --resampling no_resampling --sample-size 10000

# Plot results already saved in the results CSV.
python main.py --learning-curves --confusion-matrices

# Evaluate a chosen configuration on the reserved final holdout.
python main.py --model xgboost --feature raw_numeric --resampling no_resampling --final-evaluation

# Show all options and supported model/feature names.
python main.py --help
```

With no flags, all models, feature representations, and resampling strategies run on the full saved development splits. Visualization flags alone do not train models; add `--model` to train and plot together.

| Option | Purpose |
| --- | --- |
| `--models` / `--model` | Select one or more classifiers. |
| `--features` / `--feature` | Select representations such as `raw_numeric`, `pca`, or `pca_95`. |
| `--resampling` | Choose `no_resampling`, `undersample_majority`, or `oversample_minority`. |
| `--sample-size N` | Sample the training pool, then split it 60/10/20/10; the last 10% is withheld. |
| `--no-background [SPLIT ...]` | Remove Background rows from `train`, `val`, `test`, or `final_test`; flag alone selects all loaded splits. |
| `--model-params JSON` | Set model parameters by model name; pipeline parameters use `classifier__PARAM`. |
| `--early-stopping-rounds N` | XGBoost patience (default 20; 0 disables). |
| `--data PATH` | Source CSV for `--init` (default `data/ALLFLOWMETER_HIKARI2021.csv`). |
| `--splits-dir PATH` | Saved split directory (default `data/splits`). |
| `--results PATH` | Results CSV (default `evaluation_results.csv`). |
| `--figures PATH` | Plot directory (default `figures`). |
| `--exploration` | Generate dataset reports and exploration plots. |
| `--learning-curves` / `--confusion-matrices` | Generate the requested plots. |

`--init` writes the split CSVs and exits; rerunning overwrites them. Normal runs use train.csv/val.csv/development.csv. `--final-evaluation` requires explicit models, disallows sampling and exploration, and uses `test.csv`, writing to `final_evaluation_results.csv` by default. Reserve it for your final chosen settings.

Only CSV filenames have changed. Internal split roles and `--no-background` selectors remain unchanged: `test` selects development.csv and `final_test` selects the reserved test.csv.
