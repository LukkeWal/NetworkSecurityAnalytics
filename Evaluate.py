"""
Evaluate can take a trained classifier and a dataset and evaluate the classifiers' performance. It will report
- accuracy
- precision
- recall
- F1
- false positive rate
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from DataHandler import X_and_Y


RESULTS_FILE_PATH = Path("evaluation_results.csv")


def evaluate_classifier(
    trained_classifier: Any,
    validation_split: X_and_Y | None = None,
    results_file_path: Path | str = RESULTS_FILE_PATH,
) -> dict[str, Any]:
    """
    Evaluate one trained classifier on a validation split and append the results to a CSV file.

    If no validation split is supplied, the validation split stored on the trained classifier's dataset is used.
    """
    if validation_split is None:
        validation_split = trained_classifier.dataset.val

    x_validation = validation_split.x.select_dtypes(include="number")
    y_true = trained_classifier.label_encoder.transform(validation_split.y)
    y_pred = trained_classifier.model.predict(x_validation)

    average = "binary" if len(trained_classifier.label_encoder.classes_) == 2 else "macro"
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "classifier_name": trained_classifier.classifier_name,
        "feature_representation_name": trained_classifier.feature_representation_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "false_positive_rate": _false_positive_rate(
            y_true,
            y_pred,
            labels=list(range(len(trained_classifier.label_encoder.classes_))),
        ),
    }

    _append_results(metrics, Path(results_file_path))
    return metrics


def evaluate_classifiers(
    trained_classifiers: list[Any],
    results_file_path: Path | str = RESULTS_FILE_PATH,
) -> pd.DataFrame:
    """Evaluate multiple trained classifiers and append every result to the CSV file."""
    results = [
        evaluate_classifier(
            trained_classifier,
            results_file_path=results_file_path,
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
    results = pd.DataFrame([metrics])
    results.to_csv(
        results_file_path,
        mode="a",
        header=not results_file_path.exists(),
        index=False,
    )
