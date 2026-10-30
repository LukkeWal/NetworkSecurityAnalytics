"""
Visualization contains dataset exploration utilities, reports, and plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler

from DataHandler import DataSet

TARGET_COLUMN = "Label"
ATTACK_TYPE_COLUMN = "traffic_category"
FIGURES_PATH = Path("results/figures")
MAX_FEATURES = 100


def create_confusion_matrix_visualizations(
    results_file_path: Path | str,
    result_id: int,
    figures_path: Path | str = FIGURES_PATH,
) -> dict[str, Path]:
    """Plot default/selected matrices for one saved result."""
    row = load_result(results_file_path, result_id)
    _require_fields(row, "class_labels", "settings", "default_confusion_matrix", "confusion_matrix")
    labels = json.loads(row["class_labels"])
    settings = json.loads(row["settings"])
    matrices = {column: np.asarray(json.loads(row[column]))
                for column in ("default_confusion_matrix", "confusion_matrix")}
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for axis, column, title in zip(
        axes, ("default_confusion_matrix", "confusion_matrix"),
        (f"Default threshold: {row.get('default_threshold')}",
         f"Tuned threshold: {settings['threshold']:.4g}" if settings.get("threshold_tuned") else "Selected: default"),
    ):
        matrix = matrices[column]
        image = _plot_percentage_matrix(axis, matrix)
        axis.set_xticks(range(len(labels)), labels=labels)
        axis.set_yticks(range(len(labels)), labels=labels)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_title(title)
    figure.suptitle(
        f"Result {result_id} | {row['classifier_name']} | {row.get('feature_representation_name', '')}\n"
        f"{row.get('evaluation_split', 'test')} confusion matrices | {row.get('timestamp', '')}"
    )
    figure.colorbar(image, ax=axes, label="Percentage of actual class (%)", shrink=0.8)
    output = Path(figures_path)
    output.mkdir(parents=True, exist_ok=True)
    name = str(row["classifier_name"])
    safe_name = "".join(char if char.isalnum() or char in "_-" else "_" for char in name)
    if row.get("evaluation_split") == "final_test":
        safe_name += "_final_test"
    filename = f"result_{result_id}_{safe_name}_confusion_matrices.png"
    _save_current_figure(axes[0], output, filename)
    path = output / filename
    print(f"Saved confusion matrices to {path.resolve()}")
    return {name: path}


def plot_traffic_category_confusion(breakdown, classifier_name, settings, default_threshold, evaluation_split="test"):
    """Actual traffic categories versus binary predictions at both thresholds."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 7), layout="constrained")
    titles = [f"Default threshold: {default_threshold}",
              f"Tuned threshold: {settings['threshold']:.4g}" if settings.get("threshold_tuned") else "Selected: default"]
    predicted = ["Non-attack (0)" if label == 0 else "Attack (1)" if label == 1 else str(label)
                 for label in breakdown["predicted_labels"]]
    for axis, key, title in zip(axes, ("default", "selected"), titles):
        matrix = np.asarray(breakdown[key])
        image = _plot_percentage_matrix(axis, matrix)
        axis.set_xticks(range(len(predicted)), labels=predicted)
        axis.set_yticks(range(len(breakdown["categories"])), labels=breakdown["categories"])
        axis.set_xlabel("Predicted binary label")
        axis.set_ylabel("Actual traffic category")
        axis.set_title(title)
    figure.colorbar(image, ax=axes, label="Percentage of actual category (%)", shrink=0.8)
    figure.suptitle(f"{classifier_name}: {evaluation_split} traffic categories versus binary predictions")
    return axes[0]


def _plot_percentage_matrix(axis, matrix):
    """Colour by actual-row percentage while annotating raw sample counts."""
    matrix = np.asarray(matrix)
    totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix * 100.0, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)
    image = axis.imshow(percentages, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for (i, j), count in np.ndenumerate(matrix):
        percent = f"{percentages[i, j]:.1f}%" if totals[i, 0] else "n/a"
        axis.text(j, i, f"{int(count):,}\n({percent})", ha="center", va="center",
                  color="white" if percentages[i, j] > 50 else "black")
    return image


def create_learning_curve_visualizations(
    results_file_path: Path | str,
    result_id: int,
    figures_path: Path | str = FIGURES_PATH,
) -> dict[str, Path]:
    """Plot the recorded training and validation losses for one result."""
    row = load_result(results_file_path, result_id)
    _require_fields(row, "training_curve")
    curve = json.loads(row["training_curve"])
    title = f"Result {result_id} | {row['classifier_name']} | {row.get('feature_representation_name', '')}"
    axis = plot_learning_curve(curve, title=title)
    output = Path(figures_path)
    output.mkdir(parents=True, exist_ok=True)
    filename = f"result_{result_id}_learning_curve.png"
    _save_current_figure(axis, output, filename)
    print(f"Saved learning curve to {(output / filename).resolve()}")
    return {str(result_id): output / filename}


def create_roc_curve_visualizations(
    results_file_path: Path | str,
    result_id: int,
    figures_path: Path | str = FIGURES_PATH,
) -> dict[str, Path]:
    """Plot the recorded binary ROC curve without retraining the classifier."""
    row = load_result(results_file_path, result_id)
    _require_fields(row, "roc_curve")
    title = (f"Result {result_id} | {row['classifier_name']} | "
             f"{row.get('feature_representation_name', '')}\n"
             f"{row.get('evaluation_split', 'test')} ROC curve")
    axis = plot_roc_curve(json.loads(row["roc_curve"]), title=title)
    output = Path(figures_path)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"result_{result_id}_roc_curve.png"
    _save_current_figure(axis, output, path.name)
    print(f"Saved ROC curve to {path.resolve()}")
    return {str(result_id): path}


def create_precision_recall_curve_visualizations(
    results_file_path: Path | str,
    result_id: int,
    figures_path: Path | str = FIGURES_PATH,
    minimum_recall: float = 0.0,
) -> dict[str, Path]:
    """Plot the recorded binary precision-recall curve for one result."""
    row = load_result(results_file_path, result_id)
    _require_fields(row, "precision_recall_curve")
    title = (f"Result {result_id} | {row['classifier_name']} | "
             f"{row.get('feature_representation_name', '')}\n"
             f"{row.get('evaluation_split', 'test')} precision-recall curve")
    if minimum_recall:
        title += f" (recall {minimum_recall:g}-1)"
    axis = plot_precision_recall_curve(
        json.loads(row["precision_recall_curve"]), title=title, minimum_recall=minimum_recall,
    )
    output = Path(figures_path)
    output.mkdir(parents=True, exist_ok=True)
    suffix = "_95" if minimum_recall == 0.95 else ""
    path = output / f"result_{result_id}_precision_recall_curve{suffix}.png"
    _save_current_figure(axis, output, path.name)
    print(f"Saved precision-recall curve to {path.resolve()}")
    return {str(result_id): path}


def plot_precision_recall_curve(
    curve: dict, title: str = "Precision-recall curve", minimum_recall: float = 0.0,
):
    """Plot recall on the x-axis and precision on the y-axis."""
    precision = np.asarray(curve["precision"], dtype=float)
    recall = np.asarray(curve["recall"], dtype=float)
    if (precision.ndim != 1 or recall.ndim != 1 or len(recall) < 2
            or precision.shape != recall.shape
            or not np.isfinite(precision).all() or not np.isfinite(recall).all()
            or ((precision < 0) | (precision > 1)).any()
            or ((recall < 0) | (recall > 1)).any()
            or not ((np.diff(recall) >= 0).all() or (np.diff(recall) <= 0).all())):
        raise ValueError("Precision and recall must be equal-length arrays of at least two finite values in [0, 1], with monotonic recall.")
    if not 0 <= minimum_recall < 1:
        raise ValueError("Minimum recall must be in [0, 1).")
    full_auc = auc(recall, precision)
    if minimum_recall:
        # Keep the boundary intersection so zooming preserves the original line.
        if recall[0] > recall[-1]:
            recall, precision = recall[::-1], precision[::-1]
        start = np.searchsorted(recall, minimum_recall, side="left")
        if start == len(recall):
            raise ValueError("No recorded curve points in the requested recall range.")
        if 0 < start < len(recall) and recall[start] > minimum_recall:
            boundary_precision = np.interp(minimum_recall, recall[start - 1:start + 1],
                                           precision[start - 1:start + 1])
            precision = np.r_[boundary_precision, precision[start:]]
            recall = np.r_[minimum_recall, recall[start:]]
        else:
            recall, precision = recall[start:], precision[start:]
        padding = max(float(np.ptp(precision)) * 0.05, 0.0001)
        precision_limits = (max(0, float(precision.min()) - padding),
                            min(1, float(precision.max()) + padding))
    else:
        precision_limits = (0, 1.05)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(recall, precision, label=f"PR (full AUC = {full_auc:.4f})")
    axis.set(xlim=(minimum_recall, 1), ylim=precision_limits, xlabel="Recall",
             ylabel="Precision", title=title)
    axis.legend(loc="lower left")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return axis


def plot_roc_curve(curve: dict, title: str = "ROC curve"):
    """Plot false/true positive rates with AUC and a chance baseline."""
    fpr = np.asarray(curve["fpr"], dtype=float)
    tpr = np.asarray(curve["tpr"], dtype=float)
    if (fpr.ndim != 1 or tpr.ndim != 1 or len(fpr) < 2 or fpr.shape != tpr.shape
            or not np.isfinite(fpr).all() or not np.isfinite(tpr).all()
            or ((fpr < 0) | (fpr > 1)).any() or ((tpr < 0) | (tpr > 1)).any()
            or (np.diff(fpr) < 0).any() or (np.diff(tpr) < 0).any()):
        raise ValueError("ROC rates must be equal-length, nondecreasing arrays of at least two finite values in [0, 1].")
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(fpr, tpr, label=f"ROC (AUC = {auc(fpr, tpr):.4f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="False positive rate",
             ylabel="True positive rate", title=title)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return axis


def plot_learning_curve(curve: dict, title: str = "Learning curve"):
    """Plot a recorded ensemble's training and validation log loss."""
    steps = curve["steps"]
    if not steps or not (len(steps) == len(curve["train_loss"]) == len(curve["val_loss"])):
        raise ValueError("Learning curve steps and losses must have equal, nonzero lengths.")
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(steps, curve["train_loss"], label="Training loss")
    axis.plot(steps, curve["val_loss"], label="Validation loss")
    axis.set_xlabel({"boosting_round": "Boosting round", "tree_count": "Number of trees"}[curve["step_name"]])
    axis.set_ylabel("Log loss")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return axis


def print_dataset_information(data: DataSet) -> None:
    """Print compact information about the dataset."""
    print("Full dataset information")
    print("===================")
    print(f"Rows: {len(data.train.y):,}")
    print(f"Columns: {len(data.train.x.columns):,}")

    print("\nColumn types")
    print(_get_column_types(data).value_counts().to_string())

    print("\nLabel distribution")
    print(_get_labels(data).value_counts().sort_index().to_string())

    if _has_feature_column(data, ATTACK_TYPE_COLUMN):
        print("\nTraffic category distribution")
        print(_get_feature_column(data, ATTACK_TYPE_COLUMN).value_counts().to_string())

    missing_values = _get_missing_values(data)
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    print("\nMissing values")
    if missing_values.empty:
        print("No missing values found.")
    else:
        print(missing_values.head(MAX_FEATURES).to_string())

    numeric_features = _select_numeric_features(data, TARGET_COLUMN)
    print("\nNumeric feature summary")
    print(numeric_features.describe().transpose().head(MAX_FEATURES).to_string())


def create_exploration_visualizations(
    data: DataSet,
    sampled_data: DataSet,
    figures_path: Path | str = FIGURES_PATH,
) -> None:
    """Create and save plots for quick dataset exploration."""
    figures_path = Path(figures_path)
    figures_path.mkdir(parents=True, exist_ok=True)

    axis = plot_label_distribution(data)
    _save_current_figure(axis, figures_path, "label_distribution.png")

    if _has_feature_column(data, ATTACK_TYPE_COLUMN):
        axis = _get_feature_column(data, ATTACK_TYPE_COLUMN).value_counts().plot(
            kind="bar",
            title="Traffic category distribution",
        )
        axis.set_xlabel(ATTACK_TYPE_COLUMN)
        axis.set_ylabel("Count")
        plt.tight_layout()
        _save_current_figure(axis, figures_path, "traffic_category_distribution.png")

    axis = plot_feature_correlations(
        sampled_data,
        exclude_column=TARGET_COLUMN,
    )
    _save_current_figure(axis, figures_path, "feature_correlations.png")

    axis = plot_feature_distributions(
        sampled_data,
        exclude_column=TARGET_COLUMN,
    )
    _save_current_figure(axis, figures_path, "feature_distributions.png")

    axis = plot_feature_distributions(
        sampled_data,
        exclude_column=TARGET_COLUMN,
        log_scale=True,
    )
    _save_current_figure(axis, figures_path, "feature_distributions_log.png")

    axis = plot_outliers(
        sampled_data,
        exclude_column=TARGET_COLUMN,
    )
    _save_current_figure(axis, figures_path, "feature_outliers.png")

    tsne_2D = get_tsne_features(sampled_data)
    axis = plot_feature_embedding(tsne_2D)
    _save_current_figure(axis, figures_path, "tsne.png")

    print(f"\nSaved exploratory visualizations to {figures_path.resolve()}")


def plot_label_distribution(data: DataSet):
    """Visualize the target label distribution."""
    label_counts = _get_labels(data).value_counts().sort_index()
    axis = label_counts.plot(kind="bar", title="Label distribution")
    axis.set_xlabel(TARGET_COLUMN)
    axis.set_ylabel("Count")
    plt.tight_layout()
    return axis


def plot_feature_correlations(
    data: DataSet,
    exclude_column: str | None = None,
):
    """Visualize correlations between the strongest numeric features."""
    features = _select_high_variance_features(data, exclude_column)
    correlations = features.corr()

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlations.columns)))
    axis.set_yticks(range(len(correlations.columns)))
    axis.set_xticklabels(correlations.columns, rotation=90)
    axis.set_yticklabels(correlations.columns)
    figure.colorbar(image, ax=axis)
    axis.set_title("Feature correlations")
    plt.tight_layout()
    return axis


def plot_feature_distributions(
    data: DataSet,
    exclude_column: str | None = None,
    bins: int = 30,
    log_scale: bool = False,
):
    """Visualize distributions for the strongest numeric features with bar histograms."""
    features = _select_high_variance_features(data, exclude_column)
    columns = features.columns
    column_count = len(columns)
    grid_columns = min(4, column_count)
    grid_rows = int(np.ceil(column_count / grid_columns))

    figure, axes = plt.subplots(
        grid_rows,
        grid_columns,
        figsize=(grid_columns * 4, grid_rows * 3),
        squeeze=False,
    )

    for axis, column in zip(axes.flat, columns):
        values = features[column].dropna()
        axis.hist(values, bins=bins)
        axis.set_title(column)
        axis.set_ylabel("Count")
        if log_scale:
            axis.set_yscale("log")

    for axis in axes.flat[column_count:]:
        axis.set_visible(False)

    scale_label = "log count axis" if log_scale else "linear count axis"
    figure.suptitle(f"Feature distributions ({scale_label})")
    plt.tight_layout()
    return axes.flat[0]


def plot_outliers(
    data: DataSet,
    exclude_column: str | None = None,
):
    """Visualize outliers in numeric features with boxplots."""
    features = _select_high_variance_features(data, exclude_column)

    axis = features.plot(kind="box", figsize=(12, 6), rot=90, title="Feature outliers")
    axis.set_ylabel("Value")
    plt.tight_layout()
    return axis


def plot_feature_embedding(
    embedding: pd.DataFrame,
    color_column: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
):
    """Visualize a two-dimensional PCA, UMAP, supervised UMAP, or t-SNE result."""
    metadata_columns = {TARGET_COLUMN, ATTACK_TYPE_COLUMN}
    component_columns = [column for column in embedding.columns if column not in metadata_columns]
    x_column = x_column or component_columns[0]
    y_column = y_column or component_columns[1]
    color_column = color_column or _get_default_color_column(embedding)

    figure, axis = plt.subplots(figsize=(8, 6))
    if color_column in embedding.columns:
        for label, label_data in embedding.groupby(color_column):
            axis.scatter(label_data[x_column], label_data[y_column], label=str(label), s=10, alpha=0.7)
        axis.legend(title=color_column)
    else:
        axis.scatter(embedding[x_column], embedding[y_column], s=10, alpha=0.7)

    axis.set_xlabel(x_column)
    axis.set_ylabel(y_column)
    axis.set_title("Feature embedding")
    plt.tight_layout()
    return axis


def _select_high_variance_features(
    data: DataSet,
    exclude_column: str | None,
) -> pd.DataFrame:
    features = _select_numeric_features(data, exclude_column)
    if features.empty:
        raise ValueError("No numeric feature columns were found.")

    selected_columns = features.var().sort_values(ascending=False).head(MAX_FEATURES).index
    return features[selected_columns]


def get_tsne_features(
    data: DataSet,
    perplexity: float = 30.0,
) -> pd.DataFrame:
    """Return t-SNE feature components with the target column attached when present."""
    features = _select_numeric_features(data, TARGET_COLUMN)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    if len(features) < 2 or features.shape[1] < 2:
        raise ValueError("t-SNE requires at least two rows and two numeric features.")
    effective_perplexity = min(perplexity, len(features) - 1)
    reducer = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    components = reducer.fit_transform(scaled_features)
    embedding = pd.DataFrame(components, columns=["tsne_1", "tsne_2"], index=features.index)

    embedding[TARGET_COLUMN] = _get_labels(data).to_numpy()
    if _has_feature_column(data, ATTACK_TYPE_COLUMN):
        embedding[ATTACK_TYPE_COLUMN] = _get_feature_column(data, ATTACK_TYPE_COLUMN).to_numpy()

    return embedding


def _select_numeric_features(
    data: DataSet,
    exclude_column: str | None = TARGET_COLUMN,
) -> pd.DataFrame:
    features = _get_all_features(data)
    features = features.drop(columns=[exclude_column], errors="ignore") if exclude_column else features
    return features.select_dtypes(include="number")


def _get_all_features(data: DataSet) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.drop(columns=[TARGET_COLUMN], errors="ignore")
    return pd.concat(
        [split.x for split in [data.train, data.val, data.test]],
        ignore_index=True,
    )


def _get_labels(data: DataSet) -> pd.Series:
    if isinstance(data, pd.DataFrame):
        return data[TARGET_COLUMN]
    return pd.concat(
        [split.y for split in [data.train, data.val, data.test]],
        ignore_index=True,
    )


def _get_feature_column(data: DataSet, column: str) -> pd.Series:
    if isinstance(data, pd.DataFrame):
        return data[column]
    return pd.concat(
        [
            split.x[column]
            if column in split.x.columns
            else pd.Series([pd.NA] * len(split.x), name=column)
            for split in [data.train, data.val, data.test]
        ],
        ignore_index=True,
    )


def _has_feature_column(data: DataSet, column: str) -> bool:
    if isinstance(data, pd.DataFrame):
        return column in data.columns
    return any(column in split.x.columns for split in [data.train, data.val, data.test])


def _get_column_types(data: DataSet) -> pd.Series:
    return pd.concat(
        [
            data.train.x.dtypes,
            pd.Series({TARGET_COLUMN: data.train.y.dtype}),
        ]
    )


def _get_missing_values(data: DataSet) -> pd.Series:
    feature_missing_values = _get_all_features(data).isna().sum()
    label_missing_values = pd.Series({TARGET_COLUMN: _get_labels(data).isna().sum()})
    return pd.concat([feature_missing_values, label_missing_values])


def _save_current_figure(axis, figures_path: Path, file_name: str) -> None:
    figure = axis.get_figure()
    figure.savefig(figures_path / file_name, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _get_default_color_column(data: pd.DataFrame) -> str | None:
    if ATTACK_TYPE_COLUMN in data.columns:
        return ATTACK_TYPE_COLUMN
    if TARGET_COLUMN in data.columns:
        return TARGET_COLUMN
    return None



RESULT_PLOTS = ("confusion-matrix", "traffic-category-confusion", "learning-curve", "roc-curve",
                "precision-recall-curve", "precision-recall-curve-95")
DATASET_PLOTS = (
    "label-distribution", "traffic-category-distribution", "feature-correlations",
    "feature-distributions", "feature-distributions-log", "outliers", "tsne",
)


def load_result(results_file_path: Path | str, result_id: int) -> pd.Series:
    """Read exactly one result without changing the CSV or allocating IDs."""
    results = pd.read_csv(results_file_path)
    if "id" not in results:
        raise ValueError("Results CSV has no id column.")
    ids = pd.to_numeric(results["id"], errors="raise")
    if ids.isna().any() or (ids < 1).any() or (ids % 1 != 0).any() or ids.duplicated().any():
        raise ValueError("Result IDs must be unique positive integers.")
    selected = results.loc[ids == result_id]
    if selected.empty:
        raise ValueError(f"Result ID {result_id} was not found in {results_file_path}.")
    return selected.iloc[0]


def _require_fields(row: pd.Series, *fields: str) -> None:
    missing = [field for field in fields if pd.isna(row.get(field)) or str(row[field]).strip() == ""]
    if missing:
        raise ValueError(f"Result {row.get('id')} has no recorded {', '.join(missing)}; this plot is unavailable.")


def create_traffic_category_visualization(results_file_path, result_id, figures_path=FIGURES_PATH):
    row = load_result(results_file_path, result_id)
    _require_fields(row, "traffic_category_confusion", "settings")
    axis = plot_traffic_category_confusion(
        json.loads(row["traffic_category_confusion"]),
        f"Result {result_id} | {row['classifier_name']}",
        json.loads(row["settings"]), row.get("default_threshold"),
        evaluation_split=row.get("evaluation_split", "test"),
    )
    output = Path(figures_path)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"result_{result_id}_traffic_category_confusion_matrices.png"
    _save_current_figure(axis, output, path.name)
    print(f"Saved traffic category matrices to {path.resolve()}")
    return path


def create_dataset_visualization(data_path, plot, figures_path=FIGURES_PATH, sample_size=None):
    """Plot data from the specified CSV; no experiment reconstruction or model training."""
    from DataHandler import read_data

    data = read_data(data_path)
    if sample_size is not None and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=42)
    if data.empty:
        raise ValueError("Dataset CSV is empty.")
    if plot in ("label-distribution", "tsne") and TARGET_COLUMN not in data:
        raise ValueError(f"Dataset requires a {TARGET_COLUMN} column for {plot}.")
    if plot == "label-distribution":
        axis = plot_label_distribution(data)
    elif plot == "traffic-category-distribution":
        if ATTACK_TYPE_COLUMN not in data:
            raise ValueError(f"Dataset has no {ATTACK_TYPE_COLUMN} column.")
        axis = data[ATTACK_TYPE_COLUMN].value_counts().plot(kind="bar", title="Traffic category distribution")
        axis.set_xlabel(ATTACK_TYPE_COLUMN)
        axis.set_ylabel("Count")
    elif plot == "feature-correlations":
        axis = plot_feature_correlations(data, exclude_column=TARGET_COLUMN)
    elif plot in ("feature-distributions", "feature-distributions-log"):
        axis = plot_feature_distributions(data, exclude_column=TARGET_COLUMN,
                                          log_scale=plot.endswith("-log"))
    elif plot == "outliers":
        axis = plot_outliers(data, exclude_column=TARGET_COLUMN)
    elif plot == "tsne":
        axis = plot_feature_embedding(get_tsne_features(data))
    else:
        raise ValueError(f"Unknown dataset plot: {plot}")
    output = Path(figures_path)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{Path(data_path).stem}_{plot}.png"
    _save_current_figure(axis, output, path.name)
    print(f"Saved {plot} to {path.resolve()}")
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate a plot from a saved experiment or dataset CSV.")
    parser.add_argument("--plot", required=True, choices=RESULT_PLOTS + DATASET_PLOTS)
    parser.add_argument("--results", type=Path, help="Results CSV containing the experiment ID.")
    parser.add_argument("--id", type=int, help="Positive result ID, unique within --results.")
    parser.add_argument("--data", type=Path, help="Dataset CSV for exploration plots.")
    parser.add_argument("--output-dir", type=Path, default=FIGURES_PATH,
                        help="Plot directory (default: results/figures).")
    parser.add_argument("--sample-size", type=int, help="Sample dataset rows with seed 42 before plotting.")
    args = parser.parse_args(argv)
    if args.plot in RESULT_PLOTS:
        if args.results is None or args.id is None or args.id <= 0:
            parser.error("Experiment plots require --results and a positive --id.")
        if args.data is not None or args.sample_size is not None:
            parser.error("--data and --sample-size apply only to dataset plots.")
    else:
        if args.data is None:
            parser.error("Dataset plots require --data.")
        if args.results is not None or args.id is not None:
            parser.error("--results and --id apply only to experiment plots.")
        if args.sample_size is not None and args.sample_size <= 0:
            parser.error("--sample-size must be positive.")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.plot == "confusion-matrix":
            create_confusion_matrix_visualizations(args.results, args.id, args.output_dir)
        elif args.plot == "traffic-category-confusion":
            create_traffic_category_visualization(args.results, args.id, args.output_dir)
        elif args.plot == "learning-curve":
            create_learning_curve_visualizations(args.results, args.id, args.output_dir)
        elif args.plot == "roc-curve":
            create_roc_curve_visualizations(args.results, args.id, args.output_dir)
        elif args.plot == "precision-recall-curve":
            create_precision_recall_curve_visualizations(args.results, args.id, args.output_dir)
        elif args.plot == "precision-recall-curve-95":
            create_precision_recall_curve_visualizations(
                args.results, args.id, args.output_dir, minimum_recall=0.95,
            )
        else:
            create_dataset_visualization(args.data, args.plot, args.output_dir, args.sample_size)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Cannot generate plot: {error}") from error


if __name__ == "__main__":
    main()
