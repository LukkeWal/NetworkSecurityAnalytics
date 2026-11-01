"""
Compares the model and Suricata on one lab run: both on the encrypted traffic, and Suricata on the decrypted hop between nginx
and WordPress.

    python analyze.py <run id>
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import ModelStore  # noqa: E402
import SuricataAlerts  # noqa: E402
import ZeekFlows  # noqa: E402

ROLES = {
    "browser_attacker": "Brute force (browser)",
    "runner": "Brute force XML-RPC",
    "scanner": "Probing (WPScan)",
    "browser_benign": "Benign",
}


def main(run: str) -> None:
    base = Path("runs") / run
    results = base / "results"
    results.mkdir(exist_ok=True)
    clients = json.loads((base / "clients.json").read_text())
    split_capture(base)
    run_tools(base)

    flows = ZeekFlows.read_flows(base / "enc" / "zeek")
    flows = flows[(flows["proto"] == "tcp") & (flows["responp"] == 443)].copy()
    flows["category"] = flows["originh"].map({clients[role]: name for role, name in ROLES.items()})
    flows = flows.dropna(subset=["category"]).reset_index(drop=True)

    flows["Label"] = (flows["category"] != "Benign").astype(int)
    flows["traffic_category"] = flows["category"]
    hikari_model = PROJECT / "models" / "xgboost_flow_classifier.joblib"
    if not hikari_model.exists():  # fresh checkout: use the copy shipped with the demo
        hikari_model = PROJECT / "demo" / "assets" / "xgboost_hikari_flow_classifier.joblib"
    bundle = ModelStore.load(hikari_model)
    flows["model_flag"] = ModelStore.score(bundle, flows) >= bundle.threshold

    features = flows.drop(columns=["uid", "ts", "proto", "originh", "responh", "category", "model_flag"])
    ModelStore.train_and_save(features, results / "lab_model.joblib", drop_background_from_training=False,
                              results_file_path=results / "lab_model_training.csv")
    lab_model = ModelStore.load(results / "lab_model.joblib")
    flows["lab_model_flag"] = ModelStore.score(lab_model, flows) >= lab_model.threshold
    flows["in_lab_test"] = np.isin(np.arange(len(flows)), ModelStore.test_row_ids(features))
    encrypted = SuricataAlerts.flag_flows(flows, SuricataAlerts.read_alerts(base / "enc" / "suricata" / "eve.json"))
    flows["suricata_encrypted"] = encrypted["suricata_flag"].to_numpy()
    flows = flows.merge(decrypted_alerts(base / "plain" / "suricata" / "eve.json"), how="left", on=["originh", "originp"])
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
    sessions = flows.groupby("category")["decrypted_signature"].agg(lambda s: s.dropna().nunique())
    summary.to_csv(results / "summary.csv")
    flows.drop(columns=["uid"]).to_csv(results / "flows.csv", index=False)
    signatures = flows[flows["suricata_decrypted"]].groupby("category")["decrypted_signature"].agg(
        lambda s: s.value_counts().head(3).to_dict()
    )
    print((summary * [1, 100, 100, 100, 100, 1]).round(1).to_string())
    print("\n(lab_model is scored on its held-out 20% of flows; the other columns on all flows)")
    print("\nSuricata rules that fired on the decrypted traffic (top 3 per category):")
    print(signatures.to_string())
    plot(summary, results / "lab_detection.png")


def split_capture(base: Path, capture_name: str = "lab.pcap") -> None:
    capture = base / "captures" / capture_name
    for name, port in [("enc", 443), ("plain", 80)]:
        (base / name).mkdir(exist_ok=True)
        subprocess.run(["tcpdump", "-r", str(capture), "-w", str(base / name / f"{name}.pcap"), f"tcp port {port}"],
                       check=True, capture_output=True)


def run_tools(base: Path) -> None:
    script = PROJECT / "integration" / "run_pcap.sh"
    subprocess.run([script, base / "enc" / "enc.pcap", (base / "enc").resolve()], check=True, capture_output=True)
    subprocess.run([sys.executable, "rewrite_client_addresses.py", base / "plain" / "plain.pcap",
                    base / "plain" / "plain-clients.pcap"], check=True, capture_output=True)
    # a truncated last packet (half-written at capture time) still leaves a usable eve.json, as with Zeek
    subprocess.run([script, base / "plain" / "plain-clients.pcap", (base / "plain").resolve()],
                   capture_output=True, env={"SKIP_ZEEK": "1", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})
    if not (base / "plain" / "suricata" / "eve.json").exists():
        raise RuntimeError("Suricata produced no eve.json for the plaintext hop")


def decrypted_alerts(eve: Path) -> pd.DataFrame:
    """Alerts on the plaintext hop, attributed to the client end of the connection."""
    rows = []
    with open(eve) as eve_file:
        for line in eve_file:
            if '"event_type":"alert"' not in line:
                continue
            event = json.loads(line)
            signature = event["alert"]["signature"]
            if signature.startswith(SuricataAlerts.ENGINE_EVENT_PREFIX):
                continue
            client = (event["src_ip"], event["src_port"]) if event["dest_port"] == 80 else (event["dest_ip"], event["dest_port"])
            rows.append((*client, signature))
    table = pd.DataFrame(rows, columns=["originh", "originp", "decrypted_signature"])
    return table.groupby(["originh", "originp"], as_index=False).agg(
        decrypted_alerts=("decrypted_signature", "size"), decrypted_signature=("decrypted_signature", "first")
    )


def plot(summary: pd.DataFrame, path: Path) -> None:
    series = [("hikari_model", "HIKARI-trained model", "#b9b8b2"),
              ("lab_model", "Model trained on this lab (held-out flows)", "#2a78d6"),
              ("suricata_encrypted", "Suricata, encrypted traffic", "#eb6834"),
              ("suricata_decrypted", "Suricata, decrypted traffic", "#1baf7a")]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    positions = np.arange(len(summary))
    for offset, (column, label, color) in zip([-0.3, -0.1, 0.1, 0.3], series):
        axis.bar(positions + offset, summary[column], width=0.19, color=color, label=label)
        for position, value in zip(positions, summary[column]):
            axis.text(position + offset, value + 0.01, f"{value:.0%}", ha="center", fontsize=10)
    axis.set_xticks(positions, [f"{name}\n(n={n:,})" for name, n in zip(summary.index, summary["flows"])])
    axis.set_ylim(0, 1.12)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    axis.set_title("Lab: share of flows flagged, per traffic type", loc="left")
    figure.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1])
