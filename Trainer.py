"""
Trainer will train classifiers. It will be able to train:
    - decision trees
    - random forest
    - XGBoost
    - SVM
    - KNN
    - AdaBoost
    - MLP

Each of these classifiers can be combined with the different feature representations from PreProcessing.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from DataHandler import DataSet, X_and_Y


TARGET_COLUMN = "Label"
RANDOM_STATE = 42


@dataclass(frozen=True)
class TrainedClassifier:
    """A fitted classifier plus the label encoder used for its target values."""
    settings: dict[str, Any]
    classifier_name: str
    feature_representation_name: str
    model: object
    dataset: DataSet
    label_encoder: LabelEncoder


def get_classifiers(random_state: int = RANDOM_STATE) -> dict[str, object]:
    """Return classifiers with default settings, except fixed random seeds."""
    return {
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "xgboost": XGBClassifier(random_state=random_state),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC()),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier()),
            ]
        ),
        "adaboost": AdaBoostClassifier(random_state=random_state),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", MLPClassifier(random_state=random_state)),
            ]
        ),
    }


def compare_feature_representations_and_classifiers(
    feature_representations: dict[str, DataSet],
    exploration_sample_size: int,
) -> list[TrainedClassifier]:
    """
    Evaluate every feature representation with every classifier.

    Returns a metrics DataFrame sorted by macro F1 and, when refit=True, fitted classifiers trained on the full
    feature representation.
    """
    classifiers = get_classifiers(random_state=RANDOM_STATE)
    trained_classifiers: list[TrainedClassifier] = []

    for representation_name, representation in feature_representations.items():
        x, y, label_encoder = _split_features_and_target(representation.train)

        for classifier_name, classifier in classifiers.items():
            trained_classifier = clone(classifier).fit(x, y)
            trained_classifiers.append(
                TrainedClassifier(
                    settings={
                        "exploration_sample_size": exploration_sample_size,
                    },
                    classifier_name=classifier_name,
                    feature_representation_name=representation_name,
                    model=trained_classifier,
                    dataset=representation,
                    label_encoder=label_encoder,
                )
            )
    return trained_classifiers


def _split_features_and_target(
    split: X_and_Y,
) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    features = split.x.select_dtypes(include="number")
    label_encoder = split.label_encoder
    encoded_target = label_encoder.transform(split.y)
    target = pd.Series(encoded_target, index=split.y.index, name=split.y.name or TARGET_COLUMN)
    return features, target, label_encoder
