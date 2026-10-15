"""
Visualization contains dataset exploration utilities, reports, and plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from DataHandler import DataSet, X_and_Y

TARGET_COLUMN = "Label"
ATTACK_TYPE_COLUMN = "traffic_category"
FIGURES_PATH = Path("figures")
MAX_FEATURES = 100


def create_learning_curve_visualizations(
    results_file_path: Path | str = "evaluation_results.csv",
    figures_path: Path | str = FIGURES_PATH,
) -> dict[str, Path]:
    """Save the latest recorded XGBoost and RF curves in CSV append order.

    Includes weighted variants. Historical rows without curves are skipped.
    Returns the saved plot paths, keyed by model family.
    """
    results_file_path = Path(results_file_path)
    if not results_file_path.exists():
        print(f"No learning curves found: {results_file_path} does not exist.")
        return {}
    results = pd.read_csv(results_file_path)
    if not {"classifier_name", "training_curve"}.issubset(results.columns):
        print("No recorded learning curves found in the results CSV.")
        return {}

    figures_path = Path(figures_path)
    saved_paths = {}
    families = {
        "xgboost": ("xgboost", "xgboost_scale_pos_weight"),
        "random_forest": ("random_forest", "random_forest_balanced"),
    }
    for family, classifier_names in families.items():
        candidates = results.loc[
            results["classifier_name"].isin(classifier_names)
            & results["training_curve"].notna()
            & results["training_curve"].astype(str).str.strip().ne("")
        ]
        if candidates.empty:
            print(f"No recorded learning curve found for {family}.")
            continue
        latest = candidates.iloc[-1]
        curve = json.loads(latest["training_curve"])
        title = str(latest["classifier_name"])
        for column in ("feature_representation_name", "timestamp"):
            if column in latest and pd.notna(latest[column]):
                title += f" | {latest[column]}"
        axis = plot_learning_curve(curve, title=title)
        figures_path.mkdir(parents=True, exist_ok=True)
        filename = f"{family}_learning_curve.png"
        _save_current_figure(axis, figures_path, filename)
        saved_paths[family] = figures_path / filename
        print(f"Saved learning curve to {saved_paths[family].resolve()}")
    return saved_paths


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
    figures_path.mkdir(exist_ok=True)

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
    return pd.concat(
        [split.x for split in [data.train, data.val, data.test]],
        ignore_index=True,
    )


def _get_labels(data: DataSet) -> pd.Series:
    return pd.concat(
        [split.y for split in [data.train, data.val, data.test]],
        ignore_index=True,
    )


def _get_feature_column(data: DataSet, column: str) -> pd.Series:
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


if __name__ == "__main__":
    create_learning_curve_visualizations()
