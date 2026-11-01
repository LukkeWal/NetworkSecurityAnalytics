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

## Suricata integration, TLS lab and demo

Suricata (ET Open rules) and the trained model are run on the same recorded traffic and compared flow by flow. Zeek with
zeek-flowmeter rebuilds the flows from each pcap, so the model sees the same features as the CSV. Suricata and Zeek run on
Linux and macOS (`brew install suricata zeek`; WSL on Windows). Run everything from the repository root.

```bash
integration/setup.sh      # zeek-flowmeter and the ET Open rules (needs internet), once
```

### Demo

Replays short recorded captures through Suricata, Zeek and the model and prints which detector flagged which flow. Nothing
here is live detection, and the slices have no labels, so it shows who flags what, not who is right.

| Command | What it shows | Time |
| --- | --- | --- |
| `./demo.sh check` | Smoke test: model scoring, alert-to-flow matching and the comparison code (no datasets or IDS needed) | 5 s |
| `./demo.sh xss` | CSE-CIC-IDS2018 web attacks: Suricata alerts on the script tags, the model flags nothing | 30 s |
| `./demo.sh dos` | CSE-CIC-IDS2018 DoS: the model flags about 740 flows, Suricata alerts on 4 | 30 s |
| `./demo.sh hikari` | HIKARI-2021 HTTPS brute force: Suricata raises no alerts, the HIKARI model flags 218 of 219 flows | 20 s |
| `./demo.sh lab` | Saved TLS lab capture: Suricata sees nothing in the encrypted traffic, but flags attacks once nginx has decrypted it; the lab model flags most of them either way. No Docker | 1 min |
| `./demo.sh lab-live` | Runs the TLS lab for real (WordPress behind nginx TLS, benign and attack traffic), then analyses the capture and tears the stack down. Needs Docker Desktop and about 3 GB | 3 min |

Colours in the replay output: green = both flagged, blue = Suricata only, yellow = model only.

- The slices are in `demo/assets/` (short cuts of the CSE-CIC-IDS2018, HIKARI-2021 and lab captures). The `hikari` slice
  holds attack flows only. The `lab` slice is 70 seconds, so its per-type percentages rest on a handful of held-out flows
  and move a little between runs.
- Some replay flows were in the model's training data, so headline numbers come from the held-out evaluation.
- `lab-live` uses small volumes, so its percentages are noisier than a full run. Start Docker Desktop first.
- `ModelStore.py` saves and loads the model used by the integration and the demo; `--export` above saves the full
  `main.py` pipeline (model, preprocessing and metadata).

### HIKARI-2021

```bash
python integration/fetch_hikari.py      # CSV to data/, six pcaps to pcaps/ (about 11 GB), --csv-only skips the pcaps
for f in pcaps/*/pcap/*.pcap; do integration/run_pcap.sh $f; done
python integrate.py train
python integrate.py evaluate integration/runs
```

`train` saves the model (`--keep-background` keeps background traffic in training). `evaluate` scores the held-out test
flows of the run folders and compares them with Suricata; results go to `integration_results/`.
`run_pcap.sh` reads `HOME_NET` and `EXTERNAL_NET` (default `any`), `RULES_FILE`, `SURICATA_CONFIG` and `SKIP_ZEEK`.

On the test flows of the six pcaps the model flags 7,333 of 7,360 attacks and Suricata none: the attacks are TLS to port 443,
so the rules have no content to match. `check_duplicates.py` lists attack flows that also appear as Benign with identical features.

### CSE-CIC-IDS2018

The same comparison on CSE-CIC-IDS2018, whose attacks are mostly plaintext. The captures carry no labels, so flows from the
attacker addresses documented for each day count as attacks (see `CicFlows.py`). The model is trained on these flows.

```bash
pip install remotezip
python integration/fetch_cic_captures.py pcaps_cic
for f in pcaps_cic/*/*.pcap; do
    RULES_FILE=integration/et-open-no-engine.rules SURICATA_CONFIG=integration/suricata-alerts-only.yaml \
        integration/run_pcap.sh $f integration/runs_cic/$(basename $(dirname $f))_$(basename $f .pcap)
done
python integrate_cic.py prepare integration/runs_cic
python integrate_cic.py train
python integrate_cic.py evaluate
python integrate_cic.py unseen
python integrate_cic.py sweep
python check_generalization.py
```

`prepare` labels the flows and matches Suricata's alerts to them (`--max-per-attack` caps the flows per attack type, default 5000).
`unseen` trains without one attack type at a time; `sweep` trains the classifiers from `main.py`; `check_generalization.py`
retrains with a time-ordered split and on victim-server flows only. Results go to `integration_results_cic/`.

On the held-out flows the model flags 5,769 of 6,088 attacks (1 false alarm) and Suricata 293 (5,355 false alarms).

### TLS inspection lab

WordPress brute-force and scan traffic behind an nginx TLS terminator. A recorder in nginx's network sees both the encrypted
side (port 443) and the plain HTTP hop to WordPress (port 80), so Suricata and the model can be run on the same attacks
encrypted and decrypted.

```
attackers ---HTTPS---> nginx (TLS 1.3) ---HTTP---> WordPress + MariaDB
                          |
                       recorder (tcpdump)
```

The encrypted side goes through Zeek + zeek-flowmeter (flows for the model) and Suricata. On the plain hop,
`tls-inspection-lab/rewrite_client_addresses.py` restores each client's address and port from the headers nginx passes on.
nginx closes every connection after one response, so each request is one flow.

Needs Docker, Zeek and Suricata (`integration/setup.sh`), `tcpdump` and the packages in `requirements.txt`. The certificates
are created on the first run, and `analyze.py` uses the HIKARI model from `python integrate.py train`, or the copy in
`demo/assets/` if it has not been trained.

```bash
cd tls-inspection-lab
./lab.sh up run-a        # start the stack, install WordPress
./lab.sh run run-a       # benign browsing, browser and XML-RPC brute force, WPScan
./lab.sh stop run-a
python analyze.py run-a
./lab.sh down run-a
```

`BENIGN_ROUNDS`, `BRUTE_ATTEMPTS` and `XMLRPC_ATTEMPTS` set the traffic volume. `hikari.sh` does the same with the HIKARI
authors' own scripts (`AutoSignPost.py`, `autobrowse.py`, cloned when the image is built), their host names and their
mitmproxy setting; probing is WPScan. It uses `analyze_authorscripts.py` and `analyze_background.py`.

Share of flows flagged with the authors' scripts:

| Traffic | Flows | Suricata, encrypted | Suricata, decrypted |
| --- | --- | --- | --- |
| Brute force (browser) | 2,709 | 0% | 18.9% |
| Brute force XML-RPC | 5,145 | 0% | 0.6% |
| Probing (WPScan) | 182 | 0% | 100% |
| Benign signups | 752 | 0% | 0% |

The version with our own attack code (50,629 attack and 65,097 benign flows) gives the same picture: 80.8% of attack flows
caught decrypted, 12.3% of benign flows flagged. The generated traffic is similar to HIKARI-2021 but not the same dataset:
probing is WPScan, there is no cryptominer traffic, and the target is a single WordPress server.
