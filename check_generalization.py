"""
Retrains the CSE-CIC-IDS2018 model with stricter splits than integrate_cic.py: a time-ordered split per capture and attack type,
and victim-server flows only.

    python check_generalization.py

Run integrate_cic.py prepare first. Results go to integration_results_cic/generalization*.csv.
"""

import contextlib
import io
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import DataHandler
import ModelStore
import integrate_cic

OUTPUT_PATH = integrate_cic.OUTPUT_PATH
STRATIFIED_SPLIT = ModelStore.stratified_split


def main() -> None:
    flows = pd.read_pickle(integrate_cic.FLOWS_PATH)
    data = integrate_cic.training_view(flows)
    victim = data[flows.loc[data.index, "run"].str.contains("victim")]
    print(f"benign flows from client PCs: {((flows['Label'] == 0) & ~flows['run'].str.contains('victim')).sum():,}")

    rows = []
    for name, subset, split in [
        ("random split, all flows", data, "random"),
        ("random split, victim captures only", victim, "random"),
        ("time-block split, all flows", data, "time"),
        ("time-block split, victim captures only", victim, "time"),
    ]:
        metrics, bundle = train(subset, flows, split)
        rows.append({"split": name, "flows": len(subset), **{key: metrics[key] for key in ("precision", "recall", "false_positive_rate")}})
        print(f"{name:40s} precision {rows[-1]['precision']:.4f}  recall {rows[-1]['recall']:.4f}")
    pd.DataFrame(rows).to_csv(OUTPUT_PATH / "generalization.csv", index=False)

    held_out = victim.index[time_blocks(victim, flows) >= 0.8]
    test = flows.loc[held_out].copy()
    test["flagged"] = ModelStore.score(bundle, test) >= bundle.threshold
    per_attack = test.groupby("traffic_category").agg(test_flows=("flagged", "size"), flagged=("flagged", "mean"))
    per_attack.to_csv(OUTPUT_PATH / "generalization_per_attack.csv")
    print("\nlast 20% of each capture, victim captures only:")
    print(per_attack.round(3).to_string())


def train(data: pd.DataFrame, flows: pd.DataFrame, split: str):
    """Train with the pipeline of ModelStore, swapping in the requested split for the duration of the call."""
    if split == "time":
        ModelStore.stratified_split = lambda frame: time_split(frame, flows)
    try:
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()):
            model_path = Path(scratch) / "model.joblib"
            metrics = ModelStore.train_and_save(
                data, model_path, drop_background_from_training=False, results_file_path=Path(scratch) / "results.csv"
            )
            bundle = ModelStore.load(model_path)
    finally:
        ModelStore.stratified_split = STRATIFIED_SPLIT
    return metrics, bundle


def time_blocks(data: pd.DataFrame, flows: pd.DataFrame) -> pd.Series:
    """Position of every flow within its capture and attack type, from 0 (earliest) to 1 (latest)."""
    groups = flows.loc[data.index].groupby(["run", "traffic_category"])["ts"]
    return (groups.rank(method="first") - 1) / groups.transform("size")


def time_split(data: pd.DataFrame, flows: pd.DataFrame) -> DataHandler.DataSet:
    position = time_blocks(data, flows)
    encoder = LabelEncoder().fit(data[DataHandler.TARGET_COLUMN])
    parts = [data[position < 0.6], data[(position >= 0.6) & (position < 0.8)], data[position >= 0.8]]
    return DataHandler.DataSet(*[DataHandler._split_features_and_labels(part, encoder) for part in parts])


if __name__ == "__main__":
    main()
