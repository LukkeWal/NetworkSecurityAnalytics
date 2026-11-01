"""
DataPreprocessing allows us to alter the training data to better fit our task.
We will:
- majority class undersampling
- minority class oversampling (oversampling will be performed per attack type)
- feature transformation with PCA, UMAP, supervised UMAP
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from threading import Event, Thread
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import umap

from DataHandler import DataSet, X_and_Y, _split_features_and_labels



TARGET_COLUMN = "Label"
ATTACK_TYPE_COLUMN = "traffic_category"
RANDOM_STATE = 42
UMAP_INIT = "random"
UMAP_RANDOM_STATE = RANDOM_STATE


def preprocess_data(data: DataSet, fitted: dict | None = None) -> DataSet:
    """Return a preprocessed copy of each dataset split."""
    numeric_columns = data.train.x.select_dtypes(include="number").columns
    train_numeric = _clean_numeric_features(data.train.x, numeric_columns)
    train_medians = train_numeric.median(numeric_only=True).fillna(0)
    if fitted is not None:
        fitted.update(input_columns=numeric_columns.tolist(), initial_fill_values=train_medians)

    return DataSet(
        train=_preprocess_split(data.train, numeric_columns, train_medians),
        val=_preprocess_split(data.val, numeric_columns, train_medians),
        test=_preprocess_split(data.test, numeric_columns, train_medians),
    )


def get_numeric_features(data: pd.DataFrame, exclude_column: str | None = TARGET_COLUMN) -> pd.DataFrame:
    """Return numeric feature columns with missing and infinite values handled."""
    features = data.drop(columns=[exclude_column], errors="ignore") if exclude_column else data
    features = features.select_dtypes(include="number")
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.fillna(features.median(numeric_only=True)).fillna(0)


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


@contextmanager
def _feature_step(description: str, heartbeat_seconds: float = 30):
    """Report long operations without changing their computation."""
    started = perf_counter()
    stopped = Event()
    print(f"[features] preprocessing {description}", flush=True)

    def heartbeat():
        while not stopped.wait(heartbeat_seconds):
            print(f"[features] still working on {description}: {perf_counter() - started:.0f}s elapsed", flush=True)

    reporter = Thread(target=heartbeat, daemon=True)
    reporter.start()
    try:
        yield
    except BaseException:
        stopped.set()
        reporter.join()
        print(f"[features] interrupted/failed {description}: {perf_counter() - started:.2f}s", flush=True)
        raise
    else:
        stopped.set()
        reporter.join()


def get_all_feature_representations(
    data: DataSet, selected: list[str] | None = None,
    evaluation_split_name: str = "evaluation", report_progress: bool = True,
    fitted: dict | None = None,
) -> dict[str, DataSet]:
    step = _feature_step if report_progress else lambda description: nullcontext()
    features = {}
    if selected is None or "raw_numeric" in selected:
        with step(f"raw_numeric: preparing train/val/{evaluation_split_name}"):
            raw_state = {} if fitted is not None else None
            features["raw_numeric"] = _attach_traffic_categories(_get_clean_numeric_feature_dataset(data, raw_state), data)
            if fitted is not None:
                fitted["raw_numeric"] = raw_state
    if selected == ["raw_numeric"]:
        return features
    with step(f"scaling: fit train ({len(data.train.y):,} rows), transform val/{evaluation_split_name}"):
        scaled_state = {} if fitted is not None else None
        scaled = _get_scaled_feature_splits(data, scaled_state)
    labels = data.train.label_encoder.transform(data.train.y)

    reducers = {
        "pca": PCA(random_state=RANDOM_STATE),
        "pca_95": PCA(n_components=0.95, random_state=RANDOM_STATE),
    }

    for n_components in (5, 10, 20):
        reducers[f"umap_{n_components}"] = umap.UMAP(
            n_components=n_components,
            random_state=UMAP_RANDOM_STATE,
            init=UMAP_INIT,
            verbose=False,
        )
        reducers[f"umap_supervised_{n_components}"] = umap.UMAP(
            n_components=n_components,
            random_state=UMAP_RANDOM_STATE,
            init=UMAP_INIT,
            verbose=False,
        )

    for reducer_name, reducer in reducers.items():
        if selected is not None and reducer_name not in selected:
            continue
        with step(f"{reducer_name}: fit/transform train ({len(data.train.y):,} rows)"):
            if reducer_name.startswith("umap_supervised"):
                train_components = reducer.fit_transform(scaled["train"], y=labels)
            else:
                train_components = reducer.fit_transform(scaled["train"])

        train = X_and_Y(
            x=_build_component_data_frame(train_components, reducer_name, data.train.x),
            y=data.train.y,
            label_encoder=data.train.label_encoder,
        )
        with step(f"{reducer_name}: transform val ({len(data.val.y):,} rows)"):
            val_components = reducer.transform(scaled["val"])
        val = X_and_Y(
            x=_build_component_data_frame(val_components, reducer_name, data.val.x),
            y=data.val.y,
            label_encoder=data.val.label_encoder,
        )
        with step(f"{reducer_name}: transform {evaluation_split_name} ({len(data.test.y):,} rows)"):
            test_components = reducer.transform(scaled["test"])
        test = X_and_Y(
            x=_build_component_data_frame(test_components, reducer_name, data.test.x),
            y=data.test.y,
            label_encoder=data.test.label_encoder,
        )
        result = DataSet(train=train, val=val, test=test)
        features[reducer_name] = _attach_traffic_categories(result, data)
        if fitted is not None:
            fitted[reducer_name] = {**scaled_state, "reducer": reducer}
    return features


def _attach_traffic_categories(features: DataSet, source: DataSet) -> DataSet:
    """Retain string categories for evaluation; numeric model inputs exclude them."""
    for name in ("train", "val", "test"):
        original = getattr(source, name).x
        if ATTACK_TYPE_COLUMN in original.columns:
            getattr(features, name).x[ATTACK_TYPE_COLUMN] = (
                original[ATTACK_TYPE_COLUMN].astype("string").to_numpy()
            )
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


def _combine_features_and_labels(split: X_and_Y) -> pd.DataFrame:
    combined_data = split.x.copy()
    combined_data[TARGET_COLUMN] = split.y.to_numpy()
    return combined_data


def _get_scaled_feature_splits(data: DataSet, fitted: dict | None = None) -> dict[str, pd.DataFrame]:
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
    if fitted is not None:
        fitted.update(fill_values=fill_values, scaler=scaler)
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


def _get_clean_numeric_feature_dataset(data: DataSet, fitted: dict | None = None) -> DataSet:
    numeric_columns = data.train.x.select_dtypes(include="number").columns
    train_features = _clean_numeric_features(data.train.x, numeric_columns)
    fill_values = train_features.median(numeric_only=True).fillna(0)
    if fitted is not None:
        fitted.update(fill_values=fill_values)

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


RESAMPLING_STRATEGIES = {
    "no_resampling": lambda data: data,
    "undersample_majority": majority_class_undersampling,
    "oversample_minority": minority_class_oversampling,
}
