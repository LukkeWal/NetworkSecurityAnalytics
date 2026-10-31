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
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterGrid
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
    training_curve: dict[str, Any] | None = None


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


def parameter_combinations(parameters: dict[str, Any]) -> ParameterGrid:
    """Treat lists as candidate sets and scalars as fixed values."""
    for name, value in parameters.items():
        if isinstance(value, list) and not value:
            raise ValueError(f"{name} must contain at least one candidate value")
    return ParameterGrid({
        name: value if isinstance(value, list) else [value]
        for name, value in parameters.items()
    })


def count_models(
    feature_representations: dict[str, DataSet],
    model_names: list[str] | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> int:
    """Count all parameter combinations across selected models and representations."""
    return sum(
        len(parameter_combinations((model_params or {}).get(name, {})))
        for representation in feature_representations.values()
        for name in (model_names if model_names is not None
                     else get_classifiers(target=representation.train.y))
    )


def compare_feature_representations_and_classifiers(
    feature_representations: dict[str, DataSet],
    exploration_sample_size: int | None,
    resampling_strategy: str = "no_resampling",
    model_names: list[str] | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
    no_background: list[str] | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> list[TrainedClassifier]:
    """
    Train every parameter combination for every selected classifier and representation.

    Return each fitted classifier with its settings and training curve.
    """
    trained_classifiers: list[TrainedClassifier] = []
    if progress_total is None:
        progress_total = progress_offset + count_models(feature_representations, model_names, model_params)

    for representation_name, representation in feature_representations.items():
        x, y, label_encoder = _split_features_and_target(representation.train)
        classifiers = get_classifiers(random_state=RANDOM_STATE, target=y)
        if model_names is not None:
            unavailable = set(model_names) - classifiers.keys()
            if unavailable:
                raise ValueError(f"Models unavailable for this target: {sorted(unavailable)}")
            classifiers = {name: classifiers[name] for name in model_names}

        for classifier_name, (classifier, classifier_settings) in classifiers.items():
            prototype = classifier
            for parameters in parameter_combinations((model_params or {}).get(classifier_name, {})):
                classifier = clone(prototype)
                classifier.set_params(**parameters)
                start_time = perf_counter()
                trained_classifier, training_curve = _fit_with_training_curve(
                    classifier, x, y, representation.val, label_encoder
                )
                elapsed_time = perf_counter() - start_time
                print(
                    f"[timing] {progress_offset + len(trained_classifiers) + 1}/{progress_total} "
                    f"training {resampling_strategy}/{representation_name}/{classifier_name}: "
                    f"{elapsed_time:.2f}s"
                )
                trained_classifiers.append(
                    TrainedClassifier(
                        settings={
                            "exploration_sample_size": exploration_sample_size,
                            "resampling_strategy": resampling_strategy,
                            "no_background": no_background or [],
                            "training_sample_size": int(len(y)),
                            "training_class_counts": _class_counts(y),
                            "classification_mode": "binary" if len(label_encoder.classes_) == 2 else "multiclass",
                            **classifier_settings,
                            "model_params": classifier.get_params(deep=False) if not isinstance(classifier, Pipeline)
                            else classifier.named_steps["classifier"].get_params(deep=False),
                            **({
                                "best_boosting_round": int(trained_classifier.best_iteration) + 1,
                                "best_validation_log_loss": float(trained_classifier.best_score),
                                "boosting_rounds_trained": len(training_curve["steps"]),
                            } if isinstance(trained_classifier, XGBClassifier)
                            and trained_classifier.get_params().get("early_stopping_rounds") else {}),
                        },
                        classifier_name=classifier_name,
                        feature_representation_name=representation_name,
                        model=trained_classifier,
                        dataset=representation,
                        label_encoder=label_encoder,
                        training_curve=training_curve,
                    )
                )
    return trained_classifiers


def _fit_with_training_curve(
    classifier: Any,
    x: pd.DataFrame,
    y: pd.Series,
    validation_split: X_and_Y,
    label_encoder: LabelEncoder,
) -> tuple[Any, dict[str, Any] | None]:
    """Fit a model and record cumulative ensemble log loss where supported."""
    if not isinstance(classifier, (XGBClassifier, RandomForestClassifier)):
        return classifier.fit(x, y), None

    x_val = validation_split.x.select_dtypes(include="number").loc[:, x.columns]
    y_val = label_encoder.transform(validation_split.y)
    if isinstance(classifier, XGBClassifier):
        classifier.fit(x, y, eval_set=[(x, y), (x_val, y_val)], verbose=False)
        history = classifier.evals_result()
        metric = next(reversed(history["validation_1"]))
        train_loss = history["validation_0"][metric]
        val_loss = history["validation_1"][metric]
        step_name = "boosting_round"
    else:
        classifier.fit(x, y)
        labels = classifier.classes_
        train_sum = np.zeros((len(x), len(labels)), dtype=float)
        val_sum = np.zeros((len(x_val), len(labels)), dtype=float)
        train_loss, val_loss = [], []
        # Trees use the forest's encoded class order. Prefix averages reproduce
        # predict_proba for a forest containing just the first N fitted trees.
        x_array = x.to_numpy(dtype=np.float32)
        val_array = x_val.to_numpy(dtype=np.float32)
        for count, tree in enumerate(classifier.estimators_, start=1):
            train_sum += tree.predict_proba(x_array)
            val_sum += tree.predict_proba(val_array)
            train_loss.append(float(log_loss(y, train_sum / count, labels=labels)))
            val_loss.append(float(log_loss(y_val, val_sum / count, labels=labels)))
        step_name = "tree_count"

    return classifier, {
        "metric": metric if isinstance(classifier, XGBClassifier) else "log_loss",
        "step_name": step_name,
        "steps": list(range(1, len(train_loss) + 1)),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }


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
    return negative_count / positive_count


def _class_counts(target: pd.Series) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in target.value_counts().sort_index().items()
    }
