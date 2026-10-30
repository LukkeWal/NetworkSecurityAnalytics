# Network Security Analytics CLI

## Setup
Place the HIKARI-2021 data at `data/ALLFLOWMETER_HIKARI2021.csv`

Run commands from the project directory. Install dependencies with 

`python -m pip install -r requirements.txt`.

Create our reproducible train/val/dev/test split

`python main.py --init`

## Train a model
`main.py` is the entry point for model training. Use the help command to view all available settings for model training

`python main.py --help`

### Results
After training a model it is evaluated and the results, as well as additional statistics from training, are saved in a
CSV file.

### Train our best model
To train and run our best performing model use:

`python --% main.py --model xgboost --features raw_numeric --resampling no_resampling --early-stopping-rounds 20 --no-background train --model-params "{\"xgboost\":{\"max_depth\":6,\"min_child_weight\":5,\"learning_rate\":0.3,\"subsample\":1.0,\"colsample_bytree\":1.0}}"`

## Visualize results
After training evaluation, statistics and results are automatically saved. These can be visualized by `Visualization.py`
Again use the help page to view which visualizations and settings are available. 

`python Visualization.py --help`
## Suricata integration and demo

See `integration/README.md` for the HIKARI-2021 and CSE-CIC-IDS2018 comparisons, `tls-inspection-lab/README.md` for the TLS lab
and `demo/README.md` for a short demo that replays recorded captures.
