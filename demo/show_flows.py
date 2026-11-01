"""
Prints the flows of one replayed capture with Suricata's alert and the model's score side by side, then what the model does as a
filter behind Suricata. Needs the run folder made by demo/replay.sh.

    python demo/show_flows.py demo/run [cic|hikari]
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ModelStore  # noqa: E402
import SuricataAlerts  # noqa: E402
import ZeekFlows  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
COLOURS = {"both": "\033[32m", "Suricata only": "\033[34m", "model only": "\033[33m", "neither": "\033[2m"}
RESET, BOLD = "\033[0m", "\033[1m"


def main(run_dir: Path, model: str = "cic") -> None:
    bundle = ModelStore.load(ASSETS / f"xgboost_{model}_flow_classifier.joblib")
    flows = ZeekFlows.read_flows(run_dir / "zeek")
    alerts = SuricataAlerts.read_alerts(run_dir / "suricata" / "eve.json")
    flows["model_score"] = ModelStore.score(bundle, flows)
    flows["model_flag"] = flows["model_score"] >= bundle.threshold
    flows = SuricataAlerts.flag_flows(flows, alerts).sort_values("ts").reset_index(drop=True)
    flows["verdict"] = [
        "both" if s and m else "Suricata only" if s else "model only" if m else "neither"
        for s, m in zip(flows["suricata_flag"], flows["model_flag"])
    ]

    print(f"\n{BOLD}{len(flows)} flows, {len(alerts)} Suricata alerts, model threshold {bundle.threshold:.4f}{RESET}\n")
    print(f"{BOLD}{'time':>6}  {'source':<17}{'> dest:port':<22}{'Suricata alert':<44}{'score':>7}  verdict{RESET}")
    shown = pd.concat([flows[flows["verdict"] != "neither"].groupby(["verdict", "suricata_signature"], dropna=False).head(2),
                       flows[flows["verdict"] == "neither"].head(3)]).sort_values("ts")
    start = flows["ts"].min()
    for _, flow in shown.head(24).iterrows():
        signature = str(flow["suricata_signature"]) if flow["suricata_flag"] else "-"
        colour = COLOURS[flow["verdict"]]
        print(f"{colour}{flow['ts'] - start:6.0f}  {flow['originh']:<17}> {flow['responh'] + ':' + str(flow['responp']):<20}"
              f"{signature[:42]:<44}{flow['model_score']:6.4f}  {flow['verdict']}{RESET}")

    counts = flows["verdict"].value_counts()
    print(f"\n{BOLD}Who flagged what{RESET}")
    for name in ("both", "Suricata only", "model only", "neither"):
        print(f"  {COLOURS[name]}{name:<14}{counts.get(name, 0):>6}{RESET}")

    kept = int(counts.get("both", 0))
    removed = int(counts.get("Suricata only", 0))
    print(f"\n{BOLD}Model as a filter behind Suricata{RESET}")
    print(f"  Suricata alerted on {kept + removed} flows; the filter keeps {kept} and removes {removed}.")
    print("  (No labels here, so this replay cannot say how many of the removed alerts were false alarms.\n"
          "   The labelled evaluation is in integration/README.md.)\n")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "demo/run"), sys.argv[2] if len(sys.argv) > 2 else "cic")
