"""
Runs the Suricata + model comparison on HIKARI-2021. Train once, run integration/run_pcap.sh on each pcap, then evaluate.

    python integrate.py train
    python integrate.py evaluate integration/runs
"""

import argparse
from pathlib import Path

import pandas as pd

import DataHandler
import Evaluate
import IntegrationReport
import ModelStore
import SuricataAlerts
import ZeekFlows

DEFAULT_OUTPUT_PATH = Path("integration_results")


def load_run(run_dir: Path, bundle: ModelStore.ModelBundle, labelled: pd.DataFrame, test_rows) -> pd.DataFrame:
    """Score one pcap's rebuilt flows, flag them with Suricata and keep the labelled test-split flows."""
    flows = ZeekFlows.read_flows(run_dir / "zeek")
    flows = ZeekFlows.attach_labels(flows, labelled)
    matched = flows["row_id"].notna()
    print(f"[{run_dir.name}] {len(flows)} Zeek flows, {matched.sum()} matched to the dataset")
    flows = flows[matched & flows["row_id"].isin(test_rows)].reset_index(drop=True)

    flows["ml_score"] = ModelStore.score(bundle, flows)
    flows["ml_flag"] = flows["ml_score"] >= bundle.threshold
    alerts = SuricataAlerts.read_alerts(run_dir / "suricata" / "eve.json")
    flows = SuricataAlerts.flag_flows(flows, alerts)
    flows["run"] = run_dir.name
    print(f"[{run_dir.name}] {len(flows)} test-split flows, {len(alerts)} Suricata alerts")
    return flows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare Suricata and the ML flow classifier on pcap replays.")
    parser.add_argument("--data", type=Path, default=DataHandler.DATA_FILE_PATH, help="HIKARI-2021 CSV.")
    parser.add_argument("--model", type=Path, default=ModelStore.MODEL_PATH, help="Saved model bundle.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train the XGBoost model and save it.")
    train.add_argument("--keep-background", action="store_true",
                       help="Keep background traffic in the training split (default: drop it).")
    train.add_argument("--results", type=Path, default=Evaluate.RESULTS_FILE_PATH, help="Results CSV path.")

    evaluate = commands.add_parser("evaluate", help="Score run folders and compare with Suricata.")
    evaluate.add_argument("runs", nargs="+", type=Path, help="Folders made by integration/run_pcap.sh.")
    evaluate.add_argument("--split", choices=["test", "dev"], default="test",
                          help="Which held-out split to score (default: the final unseen test split).")
    evaluate.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output directory.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    data = pd.read_csv(args.data)
    data = data.loc[:, ~data.columns.str.startswith("Unnamed:")]

    if args.command == "train":
        metrics = ModelStore.train_and_save(
            data, args.model, drop_background_from_training=not args.keep_background, results_file_path=args.results
        )
        print({name: metrics[name] for name in ("precision", "recall", "false_positive_rate")})
        return

    bundle = ModelStore.load(args.model)
    labelled = data.assign(row_id=range(len(data)))
    test_rows = ModelStore.test_row_ids(data, args.split)
    flows = pd.concat(
        [load_run(run, bundle, labelled, test_rows) for run in ZeekFlows.run_folders(args.runs)], ignore_index=True
    )
    IntegrationReport.save_report(flows, args.out)
    print(IntegrationReport.agreement_table(flows))
    print(IntegrationReport.detector_metrics(flows).round(3))


if __name__ == "__main__":
    main()
