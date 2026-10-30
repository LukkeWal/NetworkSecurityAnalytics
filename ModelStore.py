"""
Trains the final binary classifier and saves it with its feature columns, training medians and tuned threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import DataHandler
import Evaluate
import PreProcessing
import Trainer
from DataHandler import DataSet, X_and_Y

MODEL_PATH = Path("models") / "xgboost_flow_classifier.joblib"
ATTACK_TYPE_COLUMN = "traffic_category"
BACKGROUND_CATEGORY = "Background"


@dataclass(frozen=True)
class ModelBundle:
    """A fitted classifier plus what is needed to feed it flows from outside the dataset."""
    model: object
    feature_columns: list[str]
    fill_values: pd.Series
    threshold: float


def train_and_save(
    data: pd.DataFrame,
    model_path: Path | str = MODEL_PATH,
    drop_background_from_training: bool = True,
    results_file_path: Path | str = Evaluate.RESULTS_FILE_PATH,
) -> dict:
    """Train XGBoost on raw numeric features, tune its threshold on validation data and save it."""
    split = DataHandler.get_stratified_split(data)
    train = split.train
    if drop_background_from_training:
        keep = train.x[ATTACK_TYPE_COLUMN] != BACKGROUND_CATEGORY
        train = X_and_Y(x=train.x[keep], y=train.y[keep], label_encoder=train.label_encoder)
    dataset = PreProcessing.preprocess_data(DataSet(train=train, val=split.val, test=split.test))

    trained = Trainer.compare_feature_representations_and_classifiers(
        {"raw_numeric": dataset}, len(data), model_names=["xgboost"]
    )[0]
    metrics = Evaluate.evaluate_classifier(trained, results_file_path=results_file_path)
    threshold = json.loads(metrics["settings"])["threshold"]
    if threshold is None:
        raise ValueError("Threshold tuning failed, nothing to save.")

    numeric_train = train.x.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    bundle = ModelBundle(
        model=trained.model,
        feature_columns=list(trained.model.feature_names_in_),
        fill_values=numeric_train.median().fillna(0),
        threshold=float(threshold),
    )
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return metrics


def load(model_path: Path | str = MODEL_PATH) -> ModelBundle:
    return joblib.load(model_path)


def score(bundle: ModelBundle, flows: pd.DataFrame) -> np.ndarray:
    """Return the attack probability for every flow, using the same cleaning as training."""
    x = flows.reindex(columns=bundle.feature_columns).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(bundle.fill_values).fillna(0)
    return bundle.model.predict_proba(x)[:, 1]


def test_row_ids(data: pd.DataFrame) -> np.ndarray:
    """Row positions in the CSV that fall in the held-out test split."""
    indexed = data.assign(row_id=np.arange(len(data)))
    return DataHandler.get_stratified_split(indexed).test.x["row_id"].to_numpy()
