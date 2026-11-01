"""
Same comparison as integrate.py on CSE-CIC-IDS2018, whose attacks are mostly plaintext. Run integration/run_pcap.sh on each capture
into integration/runs_cic/<name> first, then:

    python integrate_cic.py prepare integration/runs_cic
    python integrate_cic.py train
    python integrate_cic.py evaluate
    python integrate_cic.py unseen
    python integrate_cic.py sweep
"""

import argparse
from pathlib import Path

import pandas as pd

import CicFlows
import Evaluate
import IntegrationReport
import ModelStore
import PreProcessing
import SuricataAlerts
import Trainer
import ZeekFlows

OUTPUT_PATH = Path("integration_results_cic")
FLOWS_PATH = OUTPUT_PATH / "flows_labelled.pkl"
MODEL_PATH = Path("models") / "xgboost_cic_flow_classifier.joblib"
UNSEEN_MODEL_PATH = Path("models") / "xgboost_cic_unseen.joblib"
NOT_A_FEATURE = ["ts", "run"]


def prepare(run_dirs: list[Path], max_per_attack: int) -> pd.DataFrame:
    """Label every capture's flows and mark which of them Suricata alerted on."""
    runs = []
    for run_dir in run_dirs:
        flows = CicFlows.read_labelled(run_dir / "zeek", max_per_attack)
        alerts = SuricataAlerts.read_alerts(run_dir / "suricata" / "eve.json")
        flows = SuricataAlerts.flag_flows(flows, alerts)
        flows["run"] = run_dir.name
        print(f"[{run_dir.name}] {len(flows)} flows ({flows['Label'].sum()} attack), {len(alerts)} Suricata alerts")
        runs.append(flows)
    flows = pd.concat(runs, ignore_index=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    flows.to_pickle(FLOWS_PATH)
    return flows


def training_view(flows: pd.DataFrame) -> pd.DataFrame:
    """The columns the model may see: no timestamp, and no Suricata output."""
    return flows.drop(columns=NOT_A_FEATURE + [c for c in flows.columns if c.startswith("suricata_")])


def unseen_attack_types(flows: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Train without one attack type at a time and see how much of it the model still flags."""
    rows = []
    for category in sorted(set(flows["traffic_category"]) - {"Benign"}):
        known = (flows["traffic_category"] != category).to_numpy()
        subset = data[known].reset_index(drop=True)
        ModelStore.train_and_save(
            subset, UNSEEN_MODEL_PATH, drop_background_from_training=False,
            results_file_path=OUTPUT_PATH / "unseen_training_results.csv",
        )
        bundle = ModelStore.load(UNSEEN_MODEL_PATH)
        flagged = ModelStore.score(bundle, flows[~known]) >= bundle.threshold
        test = subset.iloc[ModelStore.test_row_ids(subset)]
        benign = test[test["Label"] == 0]
        false_alarms = ModelStore.score(bundle, benign) >= bundle.threshold
        rows.append({
            "unseen_attack": category,
            "flows": int((~known).sum()),
            "model_flags_it": flagged.mean(),
            "suricata_flags_it": flows.loc[~known, "suricata_flag"].mean(),
            "model_false_alarm_rate_on_benign": false_alarms.mean(),
        })
    return pd.DataFrame(rows)


def sweep(data: pd.DataFrame, models: list[str] | None, features: list[str]) -> pd.DataFrame:
    """Train every classifier and feature representation as main.py does, each with its threshold tuned at 95% recall."""
    split = ModelStore.stratified_split(data)
    processed = PreProcessing.preprocess_data(split)
    representations = PreProcessing.get_all_feature_representations(processed, selected=features)
    trained = Trainer.compare_feature_representations_and_classifiers(representations, len(data), model_names=models)
    return Evaluate.evaluate_classifiers(trained, results_file_path=OUTPUT_PATH / "sweep_results.csv")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare Suricata and the ML flow classifier on CSE-CIC-IDS2018.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Label the flows and run the Suricata matching.")
    prepare_parser.add_argument("runs", nargs="+", type=Path, help="Folders made by integration/run_pcap.sh.")
    prepare_parser.add_argument("--max-per-attack", type=int, default=5000, help="Flows kept per attack type per capture.")
    train_parser = commands.add_parser("train", help="Train the XGBoost model on the labelled flows.")
    train_parser.add_argument("--results", type=Path, default=OUTPUT_PATH / "training_results.csv")
    commands.add_parser("evaluate", help="Score the held-out test flows and compare with Suricata.")
    commands.add_parser("unseen", help="Leave each attack type out of training and test the model on it.")
    sweep_parser = commands.add_parser("sweep", help="Compare all of main.py's classifiers on these flows.")
    sweep_parser.add_argument("--models", nargs="+", help="Classifier names (default: all).")
    sweep_parser.add_argument("--features", nargs="+", default=["raw_numeric", "pca_95"], help="Feature representations.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(ZeekFlows.run_folders(args.runs), args.max_per_attack)
        return

    flows = pd.read_pickle(FLOWS_PATH)
    data = training_view(flows)
    if args.command == "train":
        metrics = ModelStore.train_and_save(
            data, MODEL_PATH, drop_background_from_training=False, results_file_path=args.results
        )
        print({name: metrics[name] for name in ("precision", "recall", "false_positive_rate")})
        return

    if args.command == "sweep":
        results = sweep(data, args.models, args.features)
        columns = ["classifier_name", "feature_representation_name", "precision", "recall", "f1", "false_positive_rate"]
        print(results[columns].sort_values("f1", ascending=False).round(3).to_string(index=False))
        return

    if args.command == "unseen":
        results = unseen_attack_types(flows, data)
        results.to_csv(OUTPUT_PATH / "unseen_attack_types.csv", index=False)
        print(results.round(3).to_string(index=False))
        return

    bundle = ModelStore.load(MODEL_PATH)
    test = flows.iloc[ModelStore.test_row_ids(data)].copy()
    test["ml_score"] = ModelStore.score(bundle, test)
    test["ml_flag"] = test["ml_score"] >= bundle.threshold
    IntegrationReport.save_report(test, OUTPUT_PATH)
    print(IntegrationReport.agreement_table(test))
    print(IntegrationReport.detector_metrics(test).round(3))
    print(IntegrationReport.per_category(test).round(3))


if __name__ == "__main__":
    main()
