# Network Security Analytics CLI

Run commands from the project directory. Install dependencies with `python -m pip install -r requirements.txt`.

```powershell
# Create reproducible train/val/dev/test CSVs (60/10/20/10).
python main.py --init

# Train and evaluate a selected configuration.
python main.py --model xgboost --features raw_numeric --resampling no_resampling

# Quick experiment using 10,000 rows from the training pool.
python main.py --model xgboost --features raw_numeric --resampling no_resampling --sample-size 10000

# Plot results already saved in the results CSV.
python main.py --learning-curves --confusion-matrices

# Evaluate a chosen configuration on the reserved final holdout.
python main.py --model xgboost --features raw_numeric --resampling no_resampling --final-evaluation

# Show all options and supported model/feature names.
python main.py --help
```

With no flags, all models, feature representations, and resampling strategies run on the full saved development splits. Visualization flags alone do not train models; add `--model` to train and plot together.

| Option | Purpose |
| --- | --- |
| `--models` / `--model` | Select one or more classifiers. |
| `--features` | Select representations such as `raw_numeric`, `pca`, or `pca_95`. |
| `--resampling` | Choose `no_resampling`, `undersample_majority`, or `oversample_minority`. |
| `--sample-size N` | Sample the training pool, then split it 60/10/20/10; the last 10% is withheld. |
| `--no-background [SPLIT ...]` | Remove Background rows from `train`, `val`, `dev`, or `test`; flag alone selects all loaded splits. |
| `--model-params JSON` | Set model parameters by model name; pipeline parameters use `classifier__PARAM`. |
| `--early-stopping-rounds N` | XGBoost patience (default 20; 0 disables). |
| `--exploration` | Generate dataset reports and exploration plots. |
| `--learning-curves` / `--confusion-matrices` | Generate the requested plots. |

`--init` writes the split CSVs and exits; rerunning overwrites them. Normal runs use train.csv/val.csv/dev.csv. `--final-evaluation` requires explicit models, disallows sampling and exploration, and uses `test.csv`, writing to `final_evaluation_results.csv` by default. Reserve it for your final chosen settings.

`--no-background` selectors match CSV filenames: `dev` selects dev.csv and `test` selects the reserved test.csv. Normal runs do not load the reserved test set.

Paths are fixed: source `data/ALLFLOWMETER_HIKARI2021.csv`, splits `data/splits`, results `evaluation_results.csv` (or `final_evaluation_results.csv` for final evaluation), and plots `figures`.
