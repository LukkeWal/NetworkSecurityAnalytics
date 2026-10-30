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
python presentation_figures.py
```

`train` saves the model (`--keep-background` keeps background traffic in training). `evaluate` scores the held-out test
flows of the run folders and compares them with Suricata; results go to `integration_results/`.
`run_pcap.sh` reads `HOME_NET` and `EXTERNAL_NET` (default `any`), `RULES_FILE`, `SURICATA_CONFIG` and `SKIP_ZEEK`.

On the test flows of the six pcaps the model flags 7,333 of 7,360 attacks and Suricata none: the attacks are TLS to port 443,
so the rules have no content to match. `check_duplicates.py` lists attack flows that also appear as Benign with identical features.
