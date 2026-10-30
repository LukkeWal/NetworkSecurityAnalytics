# Demo

Everything here replays **recorded** data offline. Nothing is live detection.

Set up once, from the repository root:

```bash
pip install -r requirements.txt
brew install suricata zeek        # only needed for the replay (step 2)
integration/setup.sh              # only needed for the replay: zeek-flowmeter and the ET Open rules (needs internet)
```

## 1. Results page (about 1 second, no extra tools)

```bash
python demo/build_report.py --open
```

One page with the headline numbers, the figures and the tables for the three experiments (HIKARI-2021, TLS lab,
CSE-CIC-IDS2018) and the model-as-a-filter result. It reads the CSVs and figures already in the repository.

## 2. Replay a capture through Suricata, Zeek and the model (about 40 seconds)

```bash
demo/replay.sh xss      # web attacks: Suricata alerts, the model does not flag them
demo/replay.sh dos      # DoS: the model flags 700+ flows, Suricata alerts on 4
```

The captures are short slices of CSE-CIC-IDS2018 (`demo/assets/`). The script runs Suricata with the ET Open rules and Zeek with
zeek-flowmeter on the pcap, scores every rebuilt flow with the trained model, and prints the flows with both verdicts
(blue = Suricata only, yellow = model only, green = both). The slices have no labels, so this shows who flags what, not who is right;
the labelled results are in the results page. Some of these flows were in the model's training data, so the headline
numbers come from the held-out evaluation, not from this replay.

## 3. The TLS lab (optional, needs Docker, about 10 minutes for a small run)

```bash
cd tls-inspection-lab
./lab.sh up demo
BENIGN_ROUNDS=3 BRUTE_ATTEMPTS=30 XMLRPC_ATTEMPTS=100 ./lab.sh run demo
./lab.sh stop demo
python analyze.py demo
./lab.sh down demo
```

This generates the traffic live against a WordPress site behind a TLS terminator; the analysis afterwards is offline.
The saved full run (`tls-inspection-lab/runs/full/results/`) is what the results page and the report use. Run `lab.sh up` before
the demo and have the saved results ready: the lab uses about 3 GB of Docker memory.

## Reproducing the results from scratch

See `integration/README.md` (HIKARI-2021 and CSE-CIC-IDS2018: downloads, train, evaluate) and
`tls-inspection-lab/README.md`.
