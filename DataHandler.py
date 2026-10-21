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
SPLITS_PATH = Path("data") / "splits"
TARGET_COLUMN = "Label"
TRAIN_SIZE = 0.60
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.20
FINAL_TEST_SIZE = 0.10
SPLIT_NAMES = ("train", "val", "test", "final_test")
# Keep pipeline split roles unchanged while using the requested CSV filenames.
SPLIT_FILENAMES = dict(train="train.csv", val="val.csv", test="development.csv", final_test="test.csv")
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


def read_data(path: Path | str = DATA_FILE_PATH) -> pd.DataFrame:
    """Load the CSV into a pandas DataFrame."""
    data = pd.read_csv(path)
    return data.loc[:, ~data.columns.str.startswith("Unnamed:")]


def initialize_splits(
    source_path: Path | str = DATA_FILE_PATH,
    splits_path: Path | str = SPLITS_PATH,
) -> dict[str, Path]:
    """Persist reproducible 60/10/20/10 splits stratified by binary Label.

    Re-running initialization regenerates the same splits for an unchanged source.
    """
    source_path = Path(source_path)
    splits_path = Path(splits_path)
    paths = {name: splits_path / SPLIT_FILENAMES[name] for name in SPLIT_NAMES}
    if source_path.resolve() in {path.resolve() for path in paths.values()}:
        raise ValueError("The source CSV cannot also be a split output file.")
    frames = _get_stratified_frames(read_data(source_path))
    splits_path.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        frame = frames[name]
        frame.to_csv(path, index=False)
        print(f"Saved {name}: {len(frame):,} rows to {path.resolve()}")
    return paths


def load_training_sample(
    sample_size: int | None = None, splits_path: Path | str = SPLITS_PATH,
    no_background: list[str] | None = None,
    final_evaluation: bool = False,
) -> DataSet:
    """Load all fixed splits, or internally split an explicit training-pool sample.

    Normal runs never read test.csv. Explicit final evaluation places that
    reserved holdout in DataSet.test for the existing transformation pipeline.
    Sampled runs split 60/10/20/10 and leave the internal final 10% unused.
    Sampling occurs before any imputation, scaling, or class resampling.
    """
    paths = [Path(splits_path) / SPLIT_FILENAMES[name] for name in SPLIT_NAMES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing initialized splits: {', '.join(missing)}. "
            "Run main.py --init (with the same --splits-dir) first."
        )
    if final_evaluation and sample_size is not None:
        raise ValueError("--final-evaluation cannot be combined with --sample-size")
    train = read_data(paths[0])
    if sample_size is None:
        encoder = LabelEncoder().fit(train[TARGET_COLUMN])
        dataset = DataSet(
            train=_split_features_and_labels(train, encoder),
            val=_split_features_and_labels(read_data(paths[1]), encoder),
            test=_split_features_and_labels(read_data(paths[3] if final_evaluation else paths[2]), encoder),
        )
        # Fail early if a manually edited holdout contains unknown target labels.
        encoder.transform(dataset.val.y)
        encoder.transform(dataset.test.y)
    else:
        if sample_size <= 0:
            raise ValueError("--sample-size must be positive")
        if sample_size > len(train):
            raise ValueError(
                f"--sample-size {sample_size:,} exceeds the fixed training set size "
                f"of {len(train):,} rows."
            )
        try:
            if sample_size < len(train):
                sample, _ = train_test_split(
                    train, train_size=sample_size, random_state=RANDOM_STATE,
                    stratify=train[TARGET_COLUMN],
                )
            else:
                sample = train
            dataset = get_stratified_split(sample)
        except ValueError as error:
            raise ValueError(f"Cannot create a stratified sample and run splits: {error}") from error
    if no_background:
        unknown = set(no_background) - set(SPLIT_NAMES)
        if unknown:
            raise ValueError(f"Unknown --no-background splits: {sorted(unknown)}")
        loaded = {"train": dataset.train, "val": dataset.val, "test": dataset.test}
        for field, split in loaded.items():
            split_name = "final_test" if field == "test" and final_evaluation else field
            if split_name not in no_background:
                continue
            if "traffic_category" not in split.x.columns:
                raise ValueError(f"--no-background requires a traffic_category column in {split_name}")
            keep = split.x["traffic_category"].ne("Background")
            filtered = X_and_Y(
                x=split.x.loc[keep].reset_index(drop=True),
                y=split.y.loc[keep].reset_index(drop=True),
                label_encoder=split.label_encoder,
            )
            if filtered.y.empty:
                raise ValueError(f"--no-background removed all rows from {split_name}")
            if field == "train" and filtered.y.nunique() != split.y.nunique():
                raise ValueError("--no-background removed an entire training target class")
            print(f"Excluded {int((~keep).sum()):,} Background rows from {split_name}.")
            loaded[field] = filtered
        dataset = DataSet(**loaded)
    print(
        (f"Sampled {sample_size:,} of {len(train):,} fixed training rows: "
         if sample_size is not None else "Loaded full fixed splits: ") +
        f"train={len(dataset.train.y):,}, val={len(dataset.val.y):,}, "
        f"{'final_test' if final_evaluation else 'test'}={len(dataset.test.y):,}"
    )
    return dataset


def _get_stratified_frames(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Make all four disjoint splits; train boundaries match the previous seed."""
    train, remaining = train_test_split(
        data, train_size=TRAIN_SIZE, random_state=RANDOM_STATE,
        stratify=data[TARGET_COLUMN],
    )
    val, remaining = train_test_split(
        remaining, train_size=round(len(data) * VALIDATION_SIZE),
        random_state=RANDOM_STATE, stratify=remaining[TARGET_COLUMN],
    )
    test, final_test = train_test_split(
        remaining, train_size=round(len(data) * TEST_SIZE),
        random_state=RANDOM_STATE, stratify=remaining[TARGET_COLUMN],
    )
    return dict(train=train, val=val, test=test, final_test=final_test)


def get_stratified_split(data: pd.DataFrame) -> DataSet:
    """Split 60/10/20/10, returning development data only; final 10% is withheld."""
    frames = _get_stratified_frames(data)
    encoder = LabelEncoder().fit(frames["train"][TARGET_COLUMN])
    return DataSet(**{
        name: _split_features_and_labels(frames[name], encoder)
        for name in ("train", "val", "test")
    })


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
