"""Self-contained model bundles and evaluation independent of the feature cache."""

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import joblib
import pandas as pd

import DataHandler
import PreProcessing


def prepare_export_features(args):
    """Fit fresh features and retain their transformations without reading or writing caches."""
    source = DataHandler.load_training_sample(
        sample_size=args.sample_size, splits_path=DataHandler.SPLITS_PATH, no_background=args.no_background,
        multiclass=args.multiclass, final_evaluation=args.final_evaluation,
    )
    initial = {}
    prepared = PreProcessing.preprocess_data(source, fitted=initial)
    builders = PreProcessing.RESAMPLING_STRATEGIES
    features, preprocessing = {}, {}
    for strategy in args.resampling or builders:
        resampled = builders[strategy](prepared)
        states = {}
        features[strategy] = PreProcessing.get_all_feature_representations(
            resampled, selected=args.features, fitted=states,
            evaluation_split_name="test" if args.final_evaluation else "dev",
        )
        for name, state in states.items():
            preprocessing[(strategy, name)] = {**initial, **state}
    return features, preprocessing


def export_model(trained, metrics, results_file_path, export_dir, preprocessing) -> Path:
    """Write one uniquely named bundle, with JSON metadata written last."""
    bundle = Path(export_dir) / f"{Path(results_file_path).stem}_{metrics['id']}_{uuid4().hex}"
    bundle.mkdir(parents=True)
    joblib.dump({"model": trained.model, "label_encoder": trained.label_encoder},
                bundle / "model.joblib", compress=3)
    joblib.dump(preprocessing, bundle / "preprocessing.joblib", compress=3)
    settings = json.loads(metrics["settings"])
    metadata = {
        "classifier_name": trained.classifier_name,
        "feature_representation": trained.feature_representation_name,
        "input_columns": preprocessing["input_columns"],
        "feature_columns": trained.dataset.train.x.select_dtypes(include="number").columns.tolist(),
        "classification_mode": trained.settings["classification_mode"],
        "settings": settings,
        "threshold_tuned": settings["threshold_tuned"],
        "threshold": settings["threshold"],
        "benign_class_index": settings["benign_class_index"],
        "results_file": str(Path(results_file_path).resolve()),
        "result_id": metrics["id"],
    }
    (bundle / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Exported model bundle to {bundle}", flush=True)
    return bundle


def transform_features(frame, preprocessing, metadata):
    """Apply only fitted transformations, preserving the model's feature order."""
    columns = preprocessing["input_columns"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model input columns: {sorted(missing)}")
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    numeric = PreProcessing._clean_numeric_features(
        numeric, pd.Index(columns), preprocessing["initial_fill_values"])
    numeric = PreProcessing._clean_numeric_features(
        numeric, pd.Index(columns), preprocessing["fill_values"])
    if "scaler" in preprocessing:
        numeric = pd.DataFrame(preprocessing["scaler"].transform(numeric), columns=columns, index=frame.index)
    if "reducer" in preprocessing:
        numeric = pd.DataFrame(preprocessing["reducer"].transform(numeric),
                               columns=metadata["feature_columns"], index=frame.index)
    return numeric.loc[:, metadata["feature_columns"]]


def evaluate_saved_model(path, final_evaluation=False):
    """Evaluate a bundle on one raw fixed split; never fit, tune, cache or append results."""
    import Evaluate

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Saved model bundle does not exist: {path}")
    if not path.is_dir() and path.name != "metadata.json":
        raise ValueError("Specify a model bundle directory or its metadata.json")
    bundle = path if path.is_dir() else path.parent
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    artifact = joblib.load(bundle / "model.joblib")
    preprocessing = joblib.load(bundle / "preprocessing.joblib")
    split_name = "test" if final_evaluation else "dev"
    frame = DataHandler.read_data(DataHandler.SPLITS_PATH / f"{split_name}.csv")
    if split_name in metadata["settings"].get("no_background", []):
        if "traffic_category" not in frame:
            raise ValueError("Saved Background filtering requires a traffic_category column")
        frame = frame.loc[frame["traffic_category"].ne("Background")].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No rows to evaluate in {split_name}")
    if metadata["classification_mode"] == "multiclass":
        target = frame["traffic_category"].replace({"Background": "Benign"})
    else:
        target = frame[DataHandler.TARGET_COLUMN]
    features = transform_features(frame, preprocessing, metadata)
    if "traffic_category" in frame:
        features["traffic_category"] = frame["traffic_category"].astype("string")
    trained = SimpleNamespace(
        **artifact, settings=metadata["settings"], classifier_name=metadata["classifier_name"],
        feature_representation_name=metadata["feature_representation"],
    )
    threshold = {
        "threshold_tuned": metadata["threshold_tuned"], "threshold": metadata["threshold"],
        "benign_class_index": metadata["benign_class_index"],
        "validation_precision": None, "validation_recall": None, "validation_f1": None,
    }
    metrics = Evaluate.evaluate_classifier(
        trained, test_split=DataHandler.X_and_Y(features, target, artifact["label_encoder"]),
        results_file_path=None, evaluation_split=split_name, threshold_result=threshold,
    )
    print(f"Saved model: {bundle}\nEvaluation split: {split_name} ({len(frame):,} rows)")
    for name in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "roc_auc", "pr_auc"):
        if metrics[name] is not None:
            print(f"{name}: {metrics[name]:.6f}")
    print(f"confusion_matrix: {metrics['confusion_matrix']}")
    return metrics
