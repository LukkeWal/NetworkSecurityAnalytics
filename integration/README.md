## Suricata integration: HIKARI-2021

Runs Suricata (ET Open rules) and the trained model on the recorded HIKARI-2021 traffic and compares them flow by flow.
Zeek and zeek-flowmeter rebuild the flows from each pcap, so the model sees the same features as the CSV.
Suricata and Zeek run on Linux and macOS (`brew install suricata zeek`; WSL on Windows).

```bash
python integration/fetch_hikari.py      # CSV to data/, six pcaps to pcaps/ (about 11 GB), --csv-only skips the pcaps
integration/setup.sh                    # zeek-flowmeter and the ET Open rules
for f in pcaps/*/pcap/*.pcap; do integration/run_pcap.sh $f; done
python integrate.py train
python integrate.py evaluate integration/runs
```

`train` saves the model (`--keep-background` keeps background traffic in training). `evaluate` scores the held-out test
flows of the run folders and compares them with Suricata; results go to `integration_results/`.
`run_pcap.sh` reads `HOME_NET` and `EXTERNAL_NET` (default `any`), `RULES_FILE`, `SURICATA_CONFIG` and `SKIP_ZEEK`.

On the test flows of the six pcaps the model flags 7,333 of 7,360 attacks and Suricata none: the attacks are TLS to port 443,
so the rules have no content to match. `check_duplicates.py` lists attack flows that also appear as Benign with identical features.

## TLS inspection lab

`tls-inspection-lab/` generates WordPress brute-force and scan traffic behind an nginx TLS terminator and records both the
encrypted side and the plain HTTP hop behind it, to compare Suricata and the model on the same attacks encrypted and decrypted.
It reuses the modules in this directory.

From the repository root, once:

```bash
python integration/fetch_hikari.py --csv-only
python integrate.py train
integration/setup.sh
pip install scapy
```

Then see `tls-inspection-lab/README.md` (needs Docker).

## Suricata integration: CSE-CIC-IDS2018

The same comparison on CSE-CIC-IDS2018, whose attacks are mostly plaintext. The captures carry no labels, so flows from the
attacker addresses documented for each day count as attacks (see `CicFlows.py`). The model is trained on these flows.
Suricata and Zeek run on Linux and macOS (`brew install suricata zeek`; WSL on Windows).

```bash
pip install remotezip
python integration/fetch_cic_captures.py pcaps_cic
integration/setup.sh
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
The cross-dataset figure needs the HIKARI model (`python integration/fetch_hikari.py --csv-only`, then `python integrate.py train`).
