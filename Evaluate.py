"""
Evaluate can take a trained classifier and a dataset and evaluate the classifiers' performance. It will report
- accuracy
- precision
- recall
- F1
- false positive rate
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from DataHandler import X_and_Y


RESULTS_FILE_PATH = Path("evaluation_results.csv")
MINIMUM_RECALL_FOR_THRESHOLD = 0.95


def evaluate_classifier(
    trained_classifier: Any,
    validation_split: X_and_Y | None = None,
    test_split: X_and_Y | None = None,
    results_file_path: Path | str = RESULTS_FILE_PATH,
    evaluation_split: str = "test",
) -> dict[str, Any]:
    """
    Tune the decision threshold on validation data and evaluate one trained classifier on test data.

    If no splits are supplied, the validation and test splits stored on the trained classifier's dataset are used.
    """
    if validation_split is None:
        validation_split = trained_classifier.dataset.val
    if test_split is None:
        test_split = trained_classifier.dataset.test

    x_test = test_split.x.select_dtypes(include="number")
    y_test = trained_classifier.label_encoder.transform(test_split.y)
    threshold_result = _tune_threshold_for_precision_at_minimum_recall(
        trained_classifier,
        validation_split,
        minimum_recall=MINIMUM_RECALL_FOR_THRESHOLD,
    )
    binary = len(trained_classifier.label_encoder.classes_) == 2
    labels = list(range(len(trained_classifier.label_encoder.classes_)))
    scores = _get_positive_scores(trained_classifier.model, x_test) if binary else None
    default_threshold = None
    if binary and scores is not None:
        default_threshold = 0.5 if hasattr(trained_classifier.model, "predict_proba") else 0.0
        default_pred = (scores >= default_threshold).astype(int)
    else:
        default_pred = trained_classifier.model.predict(x_test)

    pr_auc = average_precision = None
    if scores is not None and len(np.unique(y_test)) == 2:
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, scores, pos_label=1)
        pr_auc = float(auc(pr_recall, pr_precision))
        average_precision = float(average_precision_score(y_test, scores, pos_label=1))

    if threshold_result["threshold_tuned"]:
        y_pred = _predict_with_threshold(
            trained_classifier.model,
            x_test,
            threshold_result["threshold"],
        )
    else:
        y_pred = default_pred

    settings = {
        **trained_classifier.settings,
        "threshold_tuned": threshold_result["threshold_tuned"],
        "threshold": threshold_result["threshold"],
        "threshold_selection_metric": f"max_precision_at_recall_at_least_{MINIMUM_RECALL_FOR_THRESHOLD}",
    }

    average = "binary" if len(trained_classifier.label_encoder.classes_) == 2 else "macro"
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "classifier_name": trained_classifier.classifier_name,
        "feature_representation_name": trained_classifier.feature_representation_name,
        "settings": json.dumps(settings),
        "evaluation_split": evaluation_split,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
        "false_positive_rate": _false_positive_rate(
            y_test,
            y_pred,
            labels=list(range(len(trained_classifier.label_encoder.classes_))),
        ),
        "validation_precision_at_recall_0_95": threshold_result["validation_precision"],
        "validation_recall_at_selected_threshold": threshold_result["validation_recall"],
        "validation_f1_at_selected_threshold": threshold_result["validation_f1"],
        "training_curve": (
            json.dumps(trained_classifier.training_curve, allow_nan=False)
            if getattr(trained_classifier, "training_curve", None) is not None
            else None
        ),
        "default_threshold": default_threshold,
        "default_accuracy": accuracy_score(y_test, default_pred),
        "default_precision": precision_score(y_test, default_pred, average=average, zero_division=0),
        "default_recall": recall_score(y_test, default_pred, average=average, zero_division=0),
        "default_f1": f1_score(y_test, default_pred, average=average, zero_division=0),
        "default_false_positive_rate": _false_positive_rate(y_test, default_pred, labels=labels),
        "pr_auc": pr_auc,
        "average_precision": average_precision,
        "class_labels": json.dumps(trained_classifier.label_encoder.classes_.tolist()),
        "confusion_matrix": json.dumps(confusion_matrix(y_test, y_pred, labels=labels).tolist()),
        "default_confusion_matrix": json.dumps(confusion_matrix(y_test, default_pred, labels=labels).tolist()),
        "traffic_category_confusion": None,
    }
    if binary and "traffic_category" in test_split.x.columns:
        categories = test_split.x["traffic_category"].astype("string").fillna("Unknown")
        names = sorted(categories.unique().tolist())
        counts = lambda predictions: [
            [int(np.sum((categories.to_numpy() == name) & (predictions == label))) for label in labels]
            for name in names
        ]
        metrics["traffic_category_confusion"] = json.dumps({
            "categories": names,
            "predicted_labels": trained_classifier.label_encoder.classes_.tolist(),
            "default": counts(default_pred),
            "selected": counts(y_pred),
        })

    _append_results(metrics, Path(results_file_path))
    print(f"\n{evaluation_split} evaluation: {trained_classifier.classifier_name}/{trained_classifier.feature_representation_name}")
    comparison = pd.DataFrame([
        {"threshold": default_threshold, **{key: metrics[f"default_{key}"] for key in
         ("accuracy", "precision", "recall", "f1", "false_positive_rate")}},
        {"threshold": threshold_result["threshold"], **{key: metrics[key] for key in
         ("accuracy", "precision", "recall", "f1", "false_positive_rate")}},
    ], index=["default", "tuned" if threshold_result["threshold_tuned"] else "selected (default)"])
    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))
    if pr_auc is not None:
        print(f"{evaluation_split} PR-AUC (trapezoidal): {pr_auc:.4f}; average precision: {average_precision:.4f}")
    print(f"Confusion matrices (rows=true, columns=predicted; labels={metrics['class_labels']}):")
    print(f"  default: {metrics['default_confusion_matrix']}")
    print(f"  selected: {metrics['confusion_matrix']}")
    return metrics


def _tune_threshold_for_precision_at_minimum_recall(
    trained_classifier: Any,
    validation_split: X_and_Y,
    minimum_recall: float,
) -> dict[str, Any]:
    x_validation = validation_split.x.select_dtypes(include="number")
    y_true = trained_classifier.label_encoder.transform(validation_split.y)
    if len(trained_classifier.label_encoder.classes_) != 2:
        return _untuned_threshold_result()

    scores = _get_positive_scores(trained_classifier.model, x_validation)
    if scores is None:
        return _untuned_threshold_result()

    precision, recall, thresholds = precision_recall_curve(y_true, scores, pos_label=1)
    if len(thresholds) == 0:
        return _untuned_threshold_result()

    candidate_indices = np.flatnonzero(recall[:-1] >= minimum_recall)
    if len(candidate_indices) == 0:
        return _untuned_threshold_result()

    best_index = max(
        candidate_indices,
        key=lambda index: (
            precision[index],
            thresholds[index],
            _f1_from_precision_recall(precision[index], recall[index]),
        ),
    )
    selected_threshold = float(thresholds[best_index])
    selected_precision = float(precision[best_index])
    selected_recall = float(recall[best_index])

    return {
        "threshold_tuned": True,
        "threshold": selected_threshold,
        "validation_precision": selected_precision,
        "validation_recall": selected_recall,
        "validation_f1": _f1_from_precision_recall(selected_precision, selected_recall),
    }


def _get_positive_scores(model: Any, x: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        positive_index = _positive_class_index(model)
        return probabilities[:, positive_index]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        if scores.ndim == 1:
            return scores
        positive_index = _positive_class_index(model)
        return scores[:, positive_index]

    return None


def _positive_class_index(model: Any) -> int:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return 1
    matches = np.flatnonzero(classes == 1)
    if len(matches) > 0:
        return int(matches[0])
    return len(classes) - 1


def _predict_with_threshold(model: Any, x: pd.DataFrame, threshold: float) -> np.ndarray:
    scores = _get_positive_scores(model, x)
    if scores is None:
        return model.predict(x)
    return (scores >= threshold).astype(int)


def _untuned_threshold_result() -> dict[str, Any]:
    return {
        "threshold_tuned": False,
        "threshold": None,
        "validation_precision": None,
        "validation_recall": None,
        "validation_f1": None,
    }


def _f1_from_precision_recall(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 2 * precision * recall / denominator if denominator else 0.0


def evaluate_classifiers(
    trained_classifiers: list[Any],
    results_file_path: Path | str = RESULTS_FILE_PATH,
    evaluation_split: str = "test",
) -> pd.DataFrame:
    """Evaluate multiple trained classifiers and append every result to the CSV file."""
    results = [
        evaluate_classifier(
            trained_classifier,
            results_file_path=results_file_path,
            evaluation_split=evaluation_split,
        )
        for trained_classifier in trained_classifiers
    ]
    return pd.DataFrame(results)


def _false_positive_rate(y_true: pd.Series, y_pred: pd.Series, labels: list[int] | None = None) -> float:
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    if len(labels) == 2:
        true_negative, false_positive, _, _ = matrix.ravel()
        denominator = false_positive + true_negative
        return false_positive / denominator if denominator else 0.0

    class_rates = []
    total = matrix.sum()
    for class_index in range(len(labels)):
        false_positive = matrix[:, class_index].sum() - matrix[class_index, class_index]
        true_negative = total - matrix[class_index, :].sum() - matrix[:, class_index].sum() + matrix[class_index, class_index]
        denominator = false_positive + true_negative
        class_rates.append(false_positive / denominator if denominator else 0.0)
    return sum(class_rates) / len(class_rates) if class_rates else 0.0


def _append_results(metrics: dict[str, Any], results_file_path: Path) -> None:
    results_file_path.parent.mkdir(parents=True, exist_ok=True)
    new_results = pd.DataFrame([metrics])
    if results_file_path.exists() and results_file_path.stat().st_size > 0:
        existing_results = pd.read_csv(results_file_path)
        results = pd.concat([existing_results, new_results], ignore_index=True, sort=False)
    else:
        results = new_results
    results.to_csv(results_file_path, index=False)
