"""Disk cache of experiment features, including all four transformed splits.

Inputs, seeds, preprocessing settings and library versions are assumed fixed.
Bump CACHE_VERSION or use --rebuild-cache when those assumptions change.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import joblib
import pandas as pd

import DataHandler
import PreProcessing

CACHE_VERSION = 1
CACHE_PATH = Path("cache/features")
FEATURE_NAMES = ["raw_numeric", "pca", "pca_95"] + [
    f"{kind}_{count}" for count in (5, 10, 20) for kind in ("umap", "umap_supervised")
]


def cache_key(strategy, representation, sample_size, multiclass, no_background):
    return {
        "version": CACHE_VERSION,
        "resampling": strategy,
        "representation": representation,
        "sample_size": sample_size,
        "multiclass": multiclass,
        "no_background": sorted(set(no_background or [])),
        **({"umap_init": PreProcessing.UMAP_INIT,
            "umap_random_state": PreProcessing.UMAP_RANDOM_STATE}
           if representation.startswith("umap") else {}),
    }


def cache_path(root, key):
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]
    return Path(root) / f"{key['resampling']}_{key['representation']}_{digest}.joblib"


def _save(path, key, splits):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically so interruption cannot leave a partially written entry.
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        joblib.dump({"key": key, "splits": splits}, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_features(sample_size=None, resampling=None, features=None, multiclass=False,
                  no_background=None, final_evaluation=False, cache_dir=CACHE_PATH,
                  rebuild=False):
    """Load or build each representation; final_evaluation only selects its output split."""
    if final_evaluation and sample_size is not None:
        raise ValueError("--final-evaluation cannot be combined with --sample-size")
    strategies = resampling or ["no_resampling", "undersample_majority", "oversample_minority"]
    representations = features or FEATURE_NAMES
    builders = {
        "no_resampling": lambda data: data,
        "undersample_majority": PreProcessing.majority_class_undersampling,
        "oversample_minority": PreProcessing.minority_class_oversampling,
    }
    prepared = None
    output = {}
    for strategy in strategies:
        output[strategy] = {}
        resampled = None
        for representation in representations:
            key = cache_key(strategy, representation, sample_size, multiclass, no_background)
            path = cache_path(cache_dir, key)
            if path.exists() and not rebuild:
                print(f"[cache] hit {strategy}/{representation}", flush=True)
                entry = joblib.load(path)
                if entry.get("key") != key or set(entry.get("splits", {})) != set(DataHandler.SPLIT_NAMES):
                    raise ValueError(f"Invalid feature cache {path}; use --rebuild-cache.")
                splits = entry["splits"]
            else:
                with PreProcessing._feature_step(f"{strategy}/{representation}"):
                    if prepared is None:
                        source = DataHandler.load_training_sample(
                            sample_size, no_background=no_background, multiclass=multiclass,
                            include_all_splits=True,
                        )
                        dev_size = len(source["dev"].y)
                        dev, heldout = source["dev"], source["test"]
                        combined = DataHandler.X_and_Y(
                            pd.concat([dev.x, heldout.x], ignore_index=True),
                            pd.concat([dev.y, heldout.y], ignore_index=True), dev.label_encoder,
                        )
                        prepared = PreProcessing.preprocess_data(DataHandler.DataSet(
                            source["train"], source["val"], combined,
                        ))
                    if resampled is None:
                        resampled = builders[strategy](prepared)
                    transformed = PreProcessing.get_all_feature_representations(
                        resampled, selected=[representation], evaluation_split_name="dev+test (combined)",
                        report_progress=False,
                    )[representation]
                    def part(start, stop):
                        split = transformed.test
                        return DataHandler.X_and_Y(
                            split.x.iloc[start:stop].reset_index(drop=True),
                            split.y.iloc[start:stop].reset_index(drop=True), split.label_encoder,
                        )
                    splits = {"train": transformed.train, "val": transformed.val,
                              "dev": part(0, dev_size), "test": part(dev_size, None)}
                    _save(path, key, splits)
            output[strategy][representation] = DataHandler.DataSet(
                splits["train"], splits["val"], splits["test" if final_evaluation else "dev"],
            )
    return output
