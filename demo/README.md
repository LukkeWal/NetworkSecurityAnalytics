# Demo

Replays recorded captures offline through Suricata, Zeek (zeek-flowmeter) and the trained model, and prints which
detector flagged which flow. Nothing here is live detection, and the slices have no labels, so it shows who flags what,
not who is right. The labelled results are in the report.

Setup once, from the repository root:

```bash
pip install -r requirements.txt
brew install suricata zeek
integration/setup.sh        # zeek-flowmeter and the ET Open rules (needs internet)
```

Then:

| Command | What it shows | Time |
| --- | --- | --- |
| `./demo.sh check` | Smoke test: model scoring, alert-to-flow matching and the comparison code (no datasets or IDS needed) | 5 s |
| `./demo.sh xss` | CSE-CIC-IDS2018 web attacks: Suricata alerts on the script tags, the model flags nothing | 30 s |
| `./demo.sh dos` | CSE-CIC-IDS2018 DoS: the model flags about 740 flows, Suricata alerts on 4 | 30 s |
| `./demo.sh hikari` | HIKARI-2021 HTTPS brute force: Suricata raises no alerts, the HIKARI model flags 218 of 219 flows | 20 s |
| `./demo.sh lab` | Saved TLS lab capture: Suricata sees nothing in the encrypted traffic, but flags attacks once nginx has decrypted it; the lab model flags most of them either way. No Docker | 1 min |
| `./demo.sh lab-live` | Runs the TLS lab for real (WordPress behind nginx TLS, benign and attack traffic), then analyses the capture and tears the stack down. Needs Docker Desktop and about 3 GB | 3 min |

Colours in the replay output: green = both flagged, blue = Suricata only, yellow = model only.

Notes:

- The slices are in `demo/assets/` (short cuts of the CSE-CIC-IDS2018, HIKARI-2021 and lab captures). The `hikari` slice
  holds attack flows only. The `lab` slice is 70 seconds, so its per-type percentages rest on a handful of held-out flows
  and move a little between runs; the stable numbers are the full-run ones in the report.
- Some replay flows were in the model's training data, so headline numbers come from the held-out evaluation.
- `lab-live` uses small volumes, so its percentages are noisier than the saved full run. Start Docker Desktop first. Lab details are in `tls-inspection-lab/README.md`.
- Training from scratch: `python main.py --init`, then the best-model command in the main README (about 20 s).
