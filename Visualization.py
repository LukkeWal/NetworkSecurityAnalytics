"""
Visualization contains dataset exploration utilities, reports, and plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_COLUMN = "Label"
ATTACK_TYPE_COLUMN = "traffic_category"
FIGURES_PATH = Path("figures")
EXPLORATION_SAMPLE_SIZE = 10_000
MAX_FEATURES = 100


def print_dataset_information(data: pd.DataFrame) -> None:
    """Print compact information about the dataset."""
    print("Dataset information")
    print("===================")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns):,}")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024 ** 2:,.2f} MB")

    print("\nColumn types")
    print(data.dtypes.value_counts().to_string())

    print("\nLabel distribution")
    print(data[TARGET_COLUMN].value_counts().sort_index().to_string())

    if ATTACK_TYPE_COLUMN in data.columns:
        print("\nTraffic category distribution")
        print(data[ATTACK_TYPE_COLUMN].value_counts().to_string())

    missing_values = data.isna().sum()
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
    data: pd.DataFrame,
    sampled_data: pd.DataFrame,
    pca_features: pd.DataFrame,
    tsne_features: pd.DataFrame,
    umap_features: pd.DataFrame | None = None,
    supervised_umap_features: pd.DataFrame | None = None,
    figures_path: Path | str = FIGURES_PATH,
) -> None:
    """Create and save plots for quick dataset exploration."""
    figures_path = Path(figures_path)
    figures_path.mkdir(exist_ok=True)

    axis = plot_label_distribution(data)
    _save_current_figure(axis, figures_path, "label_distribution.png")

    if ATTACK_TYPE_COLUMN in data.columns:
        axis = data[ATTACK_TYPE_COLUMN].value_counts().plot(
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

    axis = plot_feature_embedding(pca_features)
    _save_current_figure(axis, figures_path, "pca_features.png")

    axis = plot_feature_embedding(tsne_features)
    _save_current_figure(axis, figures_path, "tsne_features.png")

    if umap_features is not None:
        axis = plot_feature_embedding(umap_features)
        _save_current_figure(axis, figures_path, "umap_features.png")

    if supervised_umap_features is not None:
        axis = plot_feature_embedding(supervised_umap_features)
        _save_current_figure(axis, figures_path, "supervised_umap_features.png")

    print(f"\nSaved exploratory visualizations to {figures_path.resolve()}")


def plot_label_distribution(data: pd.DataFrame):
    """Visualize the target label distribution."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the DataFrame.")

    label_counts = data[TARGET_COLUMN].value_counts().sort_index()
    axis = label_counts.plot(kind="bar", title="Label distribution")
    axis.set_xlabel(TARGET_COLUMN)
    axis.set_ylabel("Count")
    plt.tight_layout()
    return axis


def plot_feature_correlations(
    data: pd.DataFrame,
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
    data: pd.DataFrame,
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
    data: pd.DataFrame,
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
    data: pd.DataFrame,
    exclude_column: str | None,
) -> pd.DataFrame:
    features = _select_numeric_features(data, exclude_column)
    if features.empty:
        raise ValueError("No numeric feature columns were found.")

    selected_columns = features.var().sort_values(ascending=False).head(MAX_FEATURES).index
    return features[selected_columns]


def _select_numeric_features(
    data: pd.DataFrame,
    exclude_column: str | None = TARGET_COLUMN,
) -> pd.DataFrame:
    features = data.drop(columns=[exclude_column], errors="ignore") if exclude_column else data
    return features.select_dtypes(include="number")


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
