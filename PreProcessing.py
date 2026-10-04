"""
DataPreprocessing allows us to alter the training data to better fit our task.
We will:
- majority class undersampling
- minority class oversampling (oversampling will be performed per attack type)
- feature transformation with PCA, UMAP, supervised UMAP, and t-SNE
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
import umap

from DataHandler import DataSet, X_and_Y



TARGET_COLUMN = "Label"
ATTACK_TYPE_COLUMN = "traffic_category"
RANDOM_STATE = 42


def preprocess_data(data: DataSet) -> DataSet:
    """Return a preprocessed copy of each dataset split."""
    numeric_columns = data.train.x.select_dtypes(include="number").columns
    train_numeric = _clean_numeric_features(data.train.x, numeric_columns)
    train_medians = train_numeric.median(numeric_only=True).fillna(0)

    return DataSet(
        train=_preprocess_split(data.train, numeric_columns, train_medians),
        val=_preprocess_split(data.val, numeric_columns, train_medians),
        test=_preprocess_split(data.test, numeric_columns, train_medians),
    )


def sample_data(
    data: DataSet,
    sample_size: int,
    random_state: int = RANDOM_STATE,
) -> DataSet:
    """Return a reproducible sample while preserving the train/val/test structure."""
    split_sizes = {
        "train": len(data.train.x),
        "val": len(data.val.x),
        "test": len(data.test.x),
    }
    total_size = sum(split_sizes.values())
    if total_size <= sample_size:
        return data

    sample_sizes = _allocate_split_sample_sizes(split_sizes, sample_size)
    return DataSet(
        train=_sample_split(data.train, sample_sizes["train"], random_state),
        val=_sample_split(data.val, sample_sizes["val"], random_state),
        test=_sample_split(data.test, sample_sizes["test"], random_state),
    )


def dataset_to_data_frame(data: DataSet) -> pd.DataFrame:
    """Combine a DataSet back into a DataFrame with the target column restored."""
    return pd.concat(
        [
            _combine_features_and_labels(data.train),
            _combine_features_and_labels(data.val),
            _combine_features_and_labels(data.test),
        ],
        ignore_index=True,
    )


def get_numeric_features(data: pd.DataFrame, exclude_column: str | None = TARGET_COLUMN) -> pd.DataFrame:
    """Return numeric feature columns with missing and infinite values handled."""
    features = data.drop(columns=[exclude_column], errors="ignore") if exclude_column else data
    features = features.select_dtypes(include="number")
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.fillna(features.median(numeric_only=True)).fillna(0)


def get_scaled_features(data: pd.DataFrame, exclude_column: str | None = TARGET_COLUMN) -> pd.DataFrame:
    """Return standardized numeric features as a DataFrame."""
    features = get_numeric_features(data, exclude_column)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return pd.DataFrame(scaled_features, columns=features.columns, index=features.index)


def majority_class_undersampling(
    data: DataSet,
    random_state: int = RANDOM_STATE,
) -> DataSet:
    """Undersample every training class to the size of the smallest training class."""
    train_data = _combine_features_and_labels(data.train)

    class_counts = train_data[TARGET_COLUMN].value_counts()
    minority_count = class_counts.min()
    sampled_data = [
        resample(
            class_data,
            replace=False,
            n_samples=minority_count,
            random_state=random_state,
        )
        for _, class_data in train_data.groupby(TARGET_COLUMN)
    ]

    sampled_train = pd.concat(sampled_data).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return DataSet(
        train=_split_features_and_labels(sampled_train, data.train.label_encoder),
        val=data.val,
        test=data.test,
    )


def minority_class_oversampling(
    data: DataSet,
    attack_type_column: str = ATTACK_TYPE_COLUMN,
    random_state: int = RANDOM_STATE,
) -> DataSet:
    """
    Oversample minority groups to the largest group size.

    If an attack type column exists, groups are balanced per attack type. This avoids treating all attacks as one
    interchangeable minority bucket.
    """
    train_data = _combine_features_and_labels(data.train)

    group_columns = [TARGET_COLUMN]
    if attack_type_column in train_data.columns:
        group_columns.append(attack_type_column)

    group_sizes = train_data.groupby(group_columns).size()
    majority_count = group_sizes.max()
    sampled_data = [
        resample(
            group_data,
            replace=len(group_data) < majority_count,
            n_samples=majority_count,
            random_state=random_state,
        )
        for _, group_data in train_data.groupby(group_columns)
    ]

    sampled_train = pd.concat(sampled_data).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return DataSet(
        train=_split_features_and_labels(sampled_train, data.train.label_encoder),
        val=data.val,
        test=data.test,
    )


def get_all_feature_representations(data: DataSet) -> dict[str, DataSet]:
    start_time = perf_counter()
    features = {
        "raw_numeric": _get_clean_numeric_feature_dataset(data),
    }
    print(f"[timing] feature representation raw_numeric: {perf_counter() - start_time:.2f}s")

    start_time = perf_counter()
    scaled = _get_scaled_feature_splits(data)
    print(f"[timing] feature scaling: {perf_counter() - start_time:.2f}s")
    labels = data.train.label_encoder.transform(data.train.y)

    reducers = {
        "pca": PCA(random_state=RANDOM_STATE),
        "pca_95": PCA(n_components=0.95, random_state=RANDOM_STATE),
    }

    for n_components in (5, 10, 20):
        reducers[f"umap_{n_components}"] = umap.UMAP(
            n_components=n_components,
            random_state=RANDOM_STATE,
        )
        reducers[f"umap_supervised_{n_components}"] = umap.UMAP(
            n_components=n_components,
            random_state=RANDOM_STATE,
        )

    for reducer_name, reducer in reducers.items():
        start_time = perf_counter()
        if reducer_name.startswith("umap_supervised"):
            train_components = reducer.fit_transform(
                scaled["train"], y=labels
            )
        else:
            train_components = reducer.fit_transform(scaled["train"])

        train = X_and_Y(
            x=_build_component_data_frame(train_components, reducer_name, data.train.x),
            y=data.train.y,
            label_encoder=data.train.label_encoder,
        )
        val = X_and_Y(
            x=_build_component_data_frame(reducer.transform(scaled["val"]), reducer_name, data.val.x),
            y=data.val.y,
            label_encoder=data.val.label_encoder,
        )
        test = X_and_Y(
            x=_build_component_data_frame(reducer.transform(scaled["test"]), reducer_name, data.test.x),
            y=data.test.y,
            label_encoder=data.test.label_encoder,
        )
        result = DataSet(train=train, val=val, test=test)
        features[reducer_name] = result
        print(f"[timing] feature representation {reducer_name}: {perf_counter() - start_time:.2f}s")
    return features


def _preprocess_split(
    split: X_and_Y,
    numeric_columns: pd.Index,
    fill_values: pd.Series,
) -> X_and_Y:
    processed_x = split.x.copy()
    processed_x.loc[:, numeric_columns] = _clean_numeric_features(
        processed_x,
        numeric_columns,
        fill_values,
    )
    return X_and_Y(
        x=processed_x.reset_index(drop=True),
        y=split.y.reset_index(drop=True),
        label_encoder=split.label_encoder,
    )


def _clean_numeric_features(
    data: pd.DataFrame,
    numeric_columns: pd.Index,
    fill_values: pd.Series | None = None,
) -> pd.DataFrame:
    features = data.loc[:, numeric_columns].replace([np.inf, -np.inf], np.nan)
    if fill_values is None:
        fill_values = features.median(numeric_only=True).fillna(0)
    return features.fillna(fill_values).fillna(0)


def _allocate_split_sample_sizes(split_sizes: dict[str, int], sample_size: int) -> dict[str, int]:
    raw_sizes = {
        split_name: sample_size * split_size / sum(split_sizes.values())
        for split_name, split_size in split_sizes.items()
    }
    sample_sizes = {
        split_name: min(split_sizes[split_name], int(np.floor(raw_size)))
        for split_name, raw_size in raw_sizes.items()
    }

    remaining = sample_size - sum(sample_sizes.values())
    split_names_by_remainder = sorted(
        split_sizes,
        key=lambda split_name: raw_sizes[split_name] - sample_sizes[split_name],
        reverse=True,
    )
    while remaining > 0:
        allocated_this_pass = False
        for split_name in split_names_by_remainder:
            if sample_sizes[split_name] >= split_sizes[split_name]:
                continue
            sample_sizes[split_name] += 1
            remaining -= 1
            allocated_this_pass = True
            if remaining == 0:
                break
        if not allocated_this_pass:
            break

    return sample_sizes


def _sample_split(split: X_and_Y, sample_size: int, random_state: int) -> X_and_Y:
    if sample_size == 0:
        return X_and_Y(
            x=split.x.iloc[0:0].reset_index(drop=True),
            y=split.y.iloc[0:0].reset_index(drop=True),
            label_encoder=split.label_encoder,
        )

    sampled_indices = split.x.sample(n=sample_size, random_state=random_state).index
    return X_and_Y(
        x=split.x.loc[sampled_indices].reset_index(drop=True),
        y=split.y.loc[sampled_indices].reset_index(drop=True),
        label_encoder=split.label_encoder,
    )


def _combine_features_and_labels(split: X_and_Y) -> pd.DataFrame:
    combined_data = split.x.copy()
    combined_data[TARGET_COLUMN] = split.y.to_numpy()
    return combined_data


def _split_features_and_labels(
    data: pd.DataFrame,
    label_encoder: LabelEncoder | None = None,
) -> X_and_Y:
    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(data[TARGET_COLUMN])

    return X_and_Y(
        x=data.drop(columns=TARGET_COLUMN).reset_index(drop=True),
        y=data[TARGET_COLUMN].reset_index(drop=True),
        label_encoder=label_encoder,
    )


def _get_scaled_feature_splits(data: DataSet) -> dict[str, pd.DataFrame]:
    numeric_columns = data.train.x.select_dtypes(include="number").columns
    if numeric_columns.empty:
        raise ValueError("No numeric feature columns were found in the training split.")

    train_features = get_numeric_features(data.train.x, exclude_column=None)
    fill_values = train_features.median(numeric_only=True).fillna(0)
    feature_splits = {
        "train": _clean_numeric_features(data.train.x, numeric_columns, fill_values),
        "val": _clean_numeric_features(data.val.x, numeric_columns, fill_values),
        "test": _clean_numeric_features(data.test.x, numeric_columns, fill_values),
    }

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(feature_splits["train"])
    return {
        "train": pd.DataFrame(scaled_train, columns=numeric_columns, index=data.train.x.index),
        "val": pd.DataFrame(
            scaler.transform(feature_splits["val"]),
            columns=numeric_columns,
            index=data.val.x.index,
        ),
        "test": pd.DataFrame(
            scaler.transform(feature_splits["test"]),
            columns=numeric_columns,
            index=data.test.x.index,
        ),
    }


def _get_clean_numeric_feature_dataset(data: DataSet) -> DataSet:
    numeric_columns = data.train.x.select_dtypes(include="number").columns
    train_features = _clean_numeric_features(data.train.x, numeric_columns)
    fill_values = train_features.median(numeric_only=True).fillna(0)

    return DataSet(
        train=X_and_Y(
            x=train_features.reset_index(drop=True),
            y=data.train.y.reset_index(drop=True),
            label_encoder=data.train.label_encoder,
        ),
        val=X_and_Y(
            x=_clean_numeric_features(data.val.x, numeric_columns, fill_values).reset_index(drop=True),
            y=data.val.y.reset_index(drop=True),
            label_encoder=data.val.label_encoder,
        ),
        test=X_and_Y(
            x=_clean_numeric_features(data.test.x, numeric_columns, fill_values).reset_index(drop=True),
            y=data.test.y.reset_index(drop=True),
            label_encoder=data.test.label_encoder,
        ),
    )


def _build_component_data_frame(
    components: np.ndarray,
    prefix: str,
    source_data: pd.DataFrame,
) -> pd.DataFrame:
    component_columns = [f"{prefix}_{index + 1}" for index in range(components.shape[1])]
    return pd.DataFrame(components, columns=component_columns, index=source_data.index)
