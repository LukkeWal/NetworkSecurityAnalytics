"""
DataHandler reads the raw data and will serve it in a usable format.
It will split the data into a stratified training, validation and testing set
"""

from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_FILE_PATH = Path("data") / "ALLFLOWMETER_HIKARI2021.csv"
TARGET_COLUMN = "Label"
TRAIN_SIZE = 0.60
VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20
RANDOM_STATE = 42


@dataclass(frozen=True)
class X_and_Y:
    """defines a dataset split with features and labels"""
    x: pd.DataFrame
    y: pd.Series
    label_encoder: LabelEncoder


@dataclass(frozen=True)
class DataSet:
    """defines a dataset split of train, validation and test sets. split into features and labels"""
    train: X_and_Y
    val: X_and_Y
    test: X_and_Y


def read_data() -> pd.DataFrame:
    """Load the CSV into a pandas DataFrame."""
    data = pd.read_csv(DATA_FILE_PATH)
    return data.loc[:, ~data.columns.str.startswith("Unnamed:")]


def get_stratified_split(
    data: pd.DataFrame,
) -> DataSet:
    """Return stratified splits with features and labels separated."""
    label_encoder = LabelEncoder()
    label_encoder.fit(data[TARGET_COLUMN])

    train_data, remaining_data = train_test_split(
        data,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_COLUMN],
    )

    validation_fraction = VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE)
    val_data, test_data = train_test_split(
        remaining_data,
        train_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=remaining_data[TARGET_COLUMN],
    )

    return DataSet(
        train=_split_features_and_labels(train_data, label_encoder),
        val=_split_features_and_labels(val_data, label_encoder),
        test=_split_features_and_labels(test_data, label_encoder),
    )


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
