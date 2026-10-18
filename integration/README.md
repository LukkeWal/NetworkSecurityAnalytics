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
python presentation_figures.py
```

`prepare` labels the flows and matches Suricata's alerts to them (`--max-per-attack` caps the flows per attack type, default 5000).
`unseen` trains without one attack type at a time; `sweep` trains the classifiers from `main.py`; `check_generalization.py`
retrains with a time-ordered split and on victim-server flows only. Results go to `integration_results_cic/`.

On the held-out flows the model flags 5,769 of 6,088 attacks (1 false alarm) and Suricata 293 (5,355 false alarms).
The cross-dataset figure needs the HIKARI model (`python integration/fetch_hikari.py --csv-only`, then `python integrate.py train`).
