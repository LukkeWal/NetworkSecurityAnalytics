"""
Quick check that the integration code works, without Suricata, Zeek or any dataset: python smoke_test.py

Loads the model shipped with the demo, scores a few made-up flows, matches them to made-up Suricata alerts and runs the
comparison from IntegrationReport. Takes a few seconds.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import IntegrationReport
import ModelStore
import SuricataAlerts

MODEL = Path(__file__).resolve().parent / "demo" / "assets" / "xgboost_cic_flow_classifier.joblib"
START = 1_700_000_000.0


def make_flows(bundle: ModelStore.ModelBundle) -> pd.DataFrame:
    """Three flows with the model's feature columns: a long busy one, a tiny one and a flow that shares its endpoints."""
    flows = pd.DataFrame(0.0, index=range(3), columns=bundle.feature_columns)
    flows["originh"] = ["10.0.0.5", "10.0.0.6", "10.0.0.5"]
    flows["originp"] = [40000, 40001, 40002]
    flows["responh"] = ["10.0.0.9", "10.0.0.9", "10.0.0.9"]
    flows["responp"] = [80, 80, 80]
    flows["ts"] = [START, START + 5, START + 600]
    flows["flow_duration"] = [2.0, 0.1, 1.0]
    flows["traffic_category"] = ["Attack", "Benign", "Benign"]
    flows["Label"] = [1, 0, 0]
    return flows


def write_eve(path: Path) -> None:
    """One alert for flow 0 (written in the reverse direction), one engine event that must be ignored."""
    def event(signature: str, src: str, src_port: int, dst: str, dst_port: int, seconds: float) -> dict:
        stamp = pd.Timestamp(START + seconds, unit="s", tz="UTC").isoformat()
        return {"timestamp": stamp, "event_type": "alert", "src_ip": src, "src_port": src_port, "dest_ip": dst,
                "dest_port": dst_port, "proto": "TCP", "alert": {"signature": signature, "category": "test"}}

    events = [
        event("ET TEST attack signature", "10.0.0.9", 80, "10.0.0.5", 40000, 1.0),
        event("SURICATA STREAM reassembly", "10.0.0.6", 40001, "10.0.0.9", 80, 5.0),
    ]
    path.write_text("\n".join(json.dumps(e, separators=(",", ":")) for e in events) + "\n")


def check(condition: bool, message: str) -> None:
    print(("ok    " if condition else "FAIL  ") + message)
    if not condition:
        sys.exit(1)


def main() -> None:
    bundle = ModelStore.load(MODEL)
    flows = make_flows(bundle)

    scores = ModelStore.score(bundle, flows)
    check(scores.shape == (3,) and np.all((scores >= 0) & (scores <= 1)), "model scores every flow with a probability")

    with tempfile.TemporaryDirectory() as folder:
        eve = Path(folder) / "eve.json"
        write_eve(eve)
        alerts = SuricataAlerts.read_alerts(eve)
    check(len(alerts) == 1, "engine events are dropped, real alerts are kept")

    flows = SuricataAlerts.flag_flows(flows, alerts)
    check(flows["suricata_flag"].tolist() == [True, False, False], "alert matches its flow whichever side is listed first")

    flows["ml_flag"] = [True, False, True]
    removed = IntegrationReport.false_alerts_removed(flows)
    check(removed["suricata_true_alerts"] == 1 and removed["true_alerts_lost"] == 0, "filter keeps a confirmed true alert")
    metrics = IntegrationReport.detector_metrics(flows)
    check(metrics.loc["ml", "recall"] == 1.0 and metrics.loc["suricata", "fp"] == 0, "detector metrics add up")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
