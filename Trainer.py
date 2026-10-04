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
from time import perf_counter
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


def get_classifiers(
    random_state: int = RANDOM_STATE,
    target: pd.Series | None = None,
) -> dict[str, tuple[object, dict[str, Any]]]:
    """Return default classifiers plus imbalance-aware variants where supported."""
    classifiers: dict[str, tuple[object, dict[str, Any]]] = {
        "decision_tree": (
            DecisionTreeClassifier(random_state=random_state),
            {"classifier_weighting": "none"},
        ),
        "decision_tree_balanced": (
            DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
            {
                "classifier_weighting": "class_weight",
                "class_weight": "balanced",
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=random_state),
            {"classifier_weighting": "none"},
        ),
        "random_forest_balanced": (
            RandomForestClassifier(random_state=random_state, class_weight="balanced"),
            {
                "classifier_weighting": "class_weight",
                "class_weight": "balanced",
            },
        ),
        "xgboost": (
            XGBClassifier(random_state=random_state),
            {"classifier_weighting": "none"},
        ),
        "svm": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", SVC()),
                ]
            ),
            {"classifier_weighting": "none"},
        ),
        "svm_balanced": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", SVC(class_weight="balanced")),
                ]
            ),
            {
                "classifier_weighting": "class_weight",
                "class_weight": "balanced",
            },
        ),
        "knn": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", KNeighborsClassifier()),
                ]
            ),
            {"classifier_weighting": "none"},
        ),
        "adaboost": (
            AdaBoostClassifier(random_state=random_state),
            {"classifier_weighting": "none"},
        ),
        "mlp": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", MLPClassifier(random_state=random_state)),
                ]
            ),
            {"classifier_weighting": "none"},
        ),
    }

    scale_pos_weight = _get_binary_scale_pos_weight(target)
    if scale_pos_weight is not None:
        classifiers["xgboost_scale_pos_weight"] = (
            XGBClassifier(
                random_state=random_state,
                scale_pos_weight=scale_pos_weight,
            ),
            {
                "classifier_weighting": "scale_pos_weight",
                "scale_pos_weight": scale_pos_weight,
            },
        )

    return classifiers


def compare_feature_representations_and_classifiers(
    feature_representations: dict[str, DataSet],
    exploration_sample_size: int,
    resampling_strategy: str = "no_resampling",
) -> list[TrainedClassifier]:
    """
    Evaluate every feature representation with every classifier.

    Returns a metrics DataFrame sorted by macro F1 and, when refit=True, fitted classifiers trained on the full
    feature representation.
    """
    trained_classifiers: list[TrainedClassifier] = []

    for representation_name, representation in feature_representations.items():
        x, y, label_encoder = _split_features_and_target(representation.train)
        classifiers = get_classifiers(random_state=RANDOM_STATE, target=y)

        for classifier_name, (classifier, classifier_settings) in classifiers.items():
            start_time = perf_counter()
            trained_classifier = clone(classifier).fit(x, y)
            elapsed_time = perf_counter() - start_time
            print(
                f"[timing] training {resampling_strategy}/{representation_name}/{classifier_name}: "
                f"{elapsed_time:.2f}s"
            )
            trained_classifiers.append(
                TrainedClassifier(
                    settings={
                        "exploration_sample_size": exploration_sample_size,
                        "resampling_strategy": resampling_strategy,
                        "training_sample_size": int(len(y)),
                        "training_class_counts": _class_counts(y),
                        **classifier_settings,
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


def _get_binary_scale_pos_weight(target: pd.Series | None) -> float | None:
    if target is None or target.nunique() != 2:
        return None

    counts = target.value_counts()
    positive_label = 1 if 1 in counts.index else counts.index.max()
    positive_count = int(counts[positive_label])
    negative_count = int(len(target) - positive_count)
    if positive_count == 0:
        return None
    return negative_count / positive_count


def _class_counts(target: pd.Series) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in target.value_counts().sort_index().items()
    }
