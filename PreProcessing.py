"""
DataPreprocessing allows us to alter the training data to better fit our task.
We will:
- majority class undersampling
- minority class oversampling (oversampling will be performed per attack type)
- feature transformation with PCA, UMAP, supervised UMAP, and t-SNE
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample


TARGET_COLUMN = "Label"
ATTACK_TYPE_COLUMN = "traffic_category"
RANDOM_STATE = 42
DEFAULT_N_COMPONENTS = 2


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a preprocessed copy of the dataset for visualization and modeling."""
    processed_data = data.copy()
    numeric_features = get_numeric_features(processed_data, TARGET_COLUMN)
    processed_data[numeric_features.columns] = numeric_features
    return processed_data


def sample_data(
    data: pd.DataFrame,
    sample_size: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Return a reproducible sample of the dataset."""
    if len(data) <= sample_size:
        return data

    return data.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def get_numeric_features(data: pd.DataFrame, exclude_column: str | None = TARGET_COLUMN) -> pd.DataFrame:
    """Return numeric feature columns with missing and infinite values handled."""
    features = data.drop(columns=[exclude_column], errors="ignore") if exclude_column else data
    features = features.select_dtypes(include="number")
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.fillna(features.median(numeric_only=True))


def get_scaled_features(data: pd.DataFrame, exclude_column: str | None = TARGET_COLUMN) -> pd.DataFrame:
    """Return standardized numeric features as a DataFrame."""
    features = get_numeric_features(data, exclude_column)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return pd.DataFrame(scaled_features, columns=features.columns, index=features.index)


def majority_class_undersampling(
    data: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Undersample every class to the size of the smallest class."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the DataFrame.")

    class_counts = data[TARGET_COLUMN].value_counts()
    minority_count = class_counts.min()
    sampled_data = [
        resample(
            class_data,
            replace=False,
            n_samples=minority_count,
            random_state=random_state,
        )
        for _, class_data in data.groupby(TARGET_COLUMN)
    ]

    return pd.concat(sampled_data).sample(frac=1, random_state=random_state).reset_index(drop=True)


def minority_class_oversampling(
    data: pd.DataFrame,
    attack_type_column: str = ATTACK_TYPE_COLUMN,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Oversample minority groups to the largest group size.

    If an attack type column exists, groups are balanced per attack type. This avoids treating all attacks as one
    interchangeable minority bucket.
    """
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the DataFrame.")

    group_columns = [TARGET_COLUMN]
    if attack_type_column in data.columns:
        group_columns.append(attack_type_column)

    group_sizes = data.groupby(group_columns).size()
    majority_count = group_sizes.max()
    sampled_data = [
        resample(
            group_data,
            replace=len(group_data) < majority_count,
            n_samples=majority_count,
            random_state=random_state,
        )
        for _, group_data in data.groupby(group_columns)
    ]

    return pd.concat(sampled_data).sample(frac=1, random_state=random_state).reset_index(drop=True)


def get_pca_features(
    data: pd.DataFrame,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> pd.DataFrame:
    """Return PCA feature components with the target column attached when present."""
    features = get_scaled_features(data, TARGET_COLUMN)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    components = pca.fit_transform(features)
    return _build_component_data_frame(components, "pca", data)


def get_umap_features(
    data: pd.DataFrame,
    n_components: int = DEFAULT_N_COMPONENTS,
    supervised: bool = False,
) -> pd.DataFrame:
    """Return UMAP feature components with optional supervision from the target labels."""
    import umap

    features = get_scaled_features(data, TARGET_COLUMN)
    reducer = umap.UMAP(n_components=n_components, random_state=RANDOM_STATE)

    labels = None
    if supervised:
        if TARGET_COLUMN not in data.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the DataFrame.")
        labels = LabelEncoder().fit_transform(data[TARGET_COLUMN])

    components = reducer.fit_transform(features, y=labels)
    prefix = "supervised_umap" if supervised else "umap"
    return _build_component_data_frame(components, prefix, data)


def get_tsne_features(
    data: pd.DataFrame,
    n_components: int = DEFAULT_N_COMPONENTS,
    perplexity: float = 30.0,
) -> pd.DataFrame:
    """Return t-SNE feature components with the target column attached when present."""
    features = get_scaled_features(data, TARGET_COLUMN)
    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=RANDOM_STATE,
        init="pca",
        learning_rate="auto",
    )
    components = reducer.fit_transform(features)
    return _build_component_data_frame(components, "tsne", data)


def _build_component_data_frame(
    components: np.ndarray,
    prefix: str,
    source_data: pd.DataFrame,
) -> pd.DataFrame:
    component_columns = [f"{prefix}_{index + 1}" for index in range(components.shape[1])]
    component_data = pd.DataFrame(components, columns=component_columns, index=source_data.index)

    if TARGET_COLUMN in source_data.columns:
        component_data[TARGET_COLUMN] = source_data[TARGET_COLUMN].values

    if ATTACK_TYPE_COLUMN in source_data.columns:
        component_data[ATTACK_TYPE_COLUMN] = source_data[ATTACK_TYPE_COLUMN].values

    return component_data
