"""
DataHandler reads the raw data and will serve it in a usable format.
It will split the data into a stratified training, validation and testing set
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_FILE_PATH = Path("data") / "ALLFLOWMETER_HIKARI2021.csv"
TARGET_COLUMN = "Label"
TRAIN_SIZE = 0.60
VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20
RANDOM_STATE = 42


def read_data() -> pd.DataFrame:
    """Load the CSV into a pandas DataFrame."""
    data = pd.read_csv(DATA_FILE_PATH)
    return data.loc[:, ~data.columns.str.startswith("Unnamed:")]


def get_stratified_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, and test DataFrames using a stratified split."""
    data = read_data()
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the CSV.")

    train_data, remaining_data = train_test_split(
        data,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_COLUMN],
    )

    validation_fraction_of_remaining = VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE)
    validation_data, test_data = train_test_split(
        remaining_data,
        train_size=validation_fraction_of_remaining,
        random_state=RANDOM_STATE,
        stratify=remaining_data[TARGET_COLUMN],
    )

    return (
        train_data.reset_index(drop=True),
        validation_data.reset_index(drop=True),
        test_data.reset_index(drop=True),
    )
