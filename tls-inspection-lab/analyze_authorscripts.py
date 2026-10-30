"""
Same as analyze.py, for the testbed that runs the HIKARI authors' own scripts (hikari.sh).

    python analyze_authorscripts.py <run id>
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402 - reuses split_capture, run_tools, decrypted_alerts, plot

ROLES = {
    "attacker_browser": "Brute force (browser)",
    "attacker_xmlrpc": "Brute force XML-RPC",
    "scanner": "Probing (WPScan)",
    "benign": "Benign",
}


def main(run: str) -> None:
    base = Path("runs") / run
    results = base / "results"
    results.mkdir(exist_ok=True)
    clients = pd.read_json(base / "clients.json", typ="series")
    analyze.split_capture(base, capture_name="server.pcap")
    analyze.run_tools(base)

    flows = analyze.ZeekFlows.read_flows(base / "enc" / "zeek")
    flows = flows[(flows["proto"] == "tcp") & (flows["responp"] == 443)].copy()
    flows["category"] = flows["originh"].map({clients[role]: name for role, name in ROLES.items()})
    flows = flows.dropna(subset=["category"]).reset_index(drop=True)

    flows["Label"] = (flows["category"] != "Benign").astype(int)
    flows["traffic_category"] = flows["category"]
    bundle = analyze.ModelStore.load(analyze.PROJECT / "models" / "xgboost_flow_classifier.joblib")
    flows["model_flag"] = analyze.ModelStore.score(bundle, flows) >= bundle.threshold

    features = flows.drop(columns=["uid", "ts", "proto", "originh", "responh", "category", "model_flag"])
    analyze.ModelStore.train_and_save(features, results / "lab_model.joblib", drop_background_from_training=False,
                                      results_file_path=results / "lab_model_training.csv")
    lab_model = analyze.ModelStore.load(results / "lab_model.joblib")
    flows["lab_model_flag"] = analyze.ModelStore.score(lab_model, flows) >= lab_model.threshold
    flows["in_lab_test"] = pd.Series(range(len(flows))).isin(analyze.ModelStore.test_row_ids(features)).to_numpy()

    encrypted = analyze.SuricataAlerts.flag_flows(flows, analyze.SuricataAlerts.read_alerts(base / "enc" / "suricata" / "eve.json"))
    flows["suricata_encrypted"] = encrypted["suricata_flag"].to_numpy()
    flows = flows.merge(analyze.decrypted_alerts(base / "plain" / "suricata" / "eve.json"), how="left", on=["originh", "originp"])
    flows["suricata_decrypted"] = flows["decrypted_alerts"].fillna(0) > 0

    held_out = flows[flows["in_lab_test"]]
    summary = flows.groupby("category").agg(
        flows=("uid", "size"),
        hikari_model=("model_flag", "mean"),
        suricata_encrypted=("suricata_encrypted", "mean"),
        suricata_decrypted=("suricata_decrypted", "mean"),
    )
    summary["lab_model"] = held_out.groupby("category")["lab_model_flag"].mean()
    summary["lab_model_test_flows"] = held_out.groupby("category").size()
    summary.to_csv(results / "summary.csv")
    flows.drop(columns=["uid"]).to_csv(results / "flows.csv", index=False)

    signatures = flows[flows["suricata_decrypted"]].groupby("category")["decrypted_signature"].agg(
        lambda s: s.value_counts().head(3).to_dict()
    )
    print((summary * [1, 100, 100, 100, 100, 1]).round(1).to_string())
    print("\n(lab_model is scored on its held-out 20% of flows; the other columns on all flows)")
    print("\nSuricata rules that fired on the decrypted traffic (top 3 per category):")
    print(signatures.to_string())
    analyze.plot(summary, results / "lab_detection.png")


if __name__ == "__main__":
    main(sys.argv[1])
