"""
Scores the background-browsing capture of an author-scripts run (the authors' autobrowse.py visiting external sites). There is no
plaintext hop for it, so only the encrypted-side comparison applies.

    python analyze_background.py <run id>
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import ModelStore  # noqa: E402
import SuricataAlerts  # noqa: E402
import ZeekFlows  # noqa: E402


def main(run: str) -> None:
    base = Path("runs") / run / "background"
    results = Path("runs") / run / "results"

    flows = ZeekFlows.read_flows(base / "zeek")
    flows = flows[(flows["proto"] == "tcp") & (flows["responp"] == 443)].reset_index(drop=True)

    bundle = ModelStore.load(PROJECT / "models" / "xgboost_flow_classifier.joblib")
    flows["hikari_model_flag"] = ModelStore.score(bundle, flows) >= bundle.threshold

    lab_model = ModelStore.load(results / "lab_model.joblib")
    flows["lab_model_flag"] = ModelStore.score(lab_model, flows) >= lab_model.threshold

    alerts = SuricataAlerts.read_alerts(base / "suricata" / "eve.json")
    flagged = SuricataAlerts.flag_flows(flows, alerts)
    flows["suricata_flag"] = flagged["suricata_flag"].to_numpy()
    flows["suricata_signature"] = flagged["suricata_signature"].to_numpy()

    print(f"background-browsing flows: {len(flows):,}  (real external sites, no plaintext hop available)")
    print(f"  destination hosts: {flows['responh'].nunique():,}")
    print(f"  hikari-trained model flags: {flows['hikari_model_flag'].mean():.1%}")
    print(f"  lab-trained model flags:    {flows['lab_model_flag'].mean():.1%}")
    print(f"  suricata (encrypted) flags: {flows['suricata_flag'].mean():.1%}  ({int(flows['suricata_flag'].sum())} flows)")
    if flows["suricata_flag"].any():
        print("  top signatures:")
        print(flows.loc[flows["suricata_flag"], "suricata_signature"].value_counts().head(5).to_string())

    flows.drop(columns=["uid"]).to_csv(results / "background_flows.csv", index=False)
    pd.Series({
        "flows": len(flows), "destinations": flows["responh"].nunique(),
        "hikari_model": flows["hikari_model_flag"].mean(), "lab_model": flows["lab_model_flag"].mean(),
        "suricata": flows["suricata_flag"].mean(),
    }).to_csv(results / "background_summary.csv")


if __name__ == "__main__":
    main(sys.argv[1])
