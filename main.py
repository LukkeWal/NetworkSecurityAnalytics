"""
main.py is the entry point of the code, and allows you to control which parts of the code is being run.
"""

import argparse
import json
from pathlib import Path

import DataHandler
import FeatureCache
import Trainer
import Evaluate
import ModelArtifacts
import PreProcessing
import pandas as pd

EXPLORATION_SAMPLE_SIZE = None

def blanket_search(feature_representations_by_resampling: dict, sample_size=EXPLORATION_SAMPLE_SIZE, models=None, model_params=None, no_background=None):
    trained_models = []
    total_models = sum(
        Trainer.count_models(representations, models, model_params)
        for representations in feature_representations_by_resampling.values()
    )
    for resampling_strategy, feature_representations in feature_representations_by_resampling.items():
        trained_models.extend(
            Trainer.compare_feature_representations_and_classifiers(
                feature_representations,
                sample_size,
                resampling_strategy=resampling_strategy,
                model_names=models,
                model_params=model_params,
                no_background=no_background,
                progress_offset=len(trained_models),
                progress_total=total_models,
            )
        )
    return trained_models

FEATURE_NAMES = FeatureCache.FEATURE_NAMES
MODEL_NAMES = list(Trainer.get_classifiers()) + ["xgboost_scale_pos_weight"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate selected network traffic classifiers.")
    parser.add_argument("--multiclass", action="store_true",
                        help="Predict Benign (including Background) or the specific attack category instead of binary labels.")
    parser.add_argument("--models", "--model", nargs="+", choices=MODEL_NAMES,
                        help="Models to train (default: all).")
    parser.add_argument("--features", nargs="+", choices=FEATURE_NAMES,
                        help="Feature representations (default: all).")
    parser.add_argument("--sample-size", "--exploration-sample-size", type=int,
                        default=EXPLORATION_SAMPLE_SIZE, help="Sample rows from train.csv and split 60/10/20/10 (last 10%% withheld); omit to use fixed development splits.")
    parser.add_argument("--resampling", nargs="+",
                        choices=list(PreProcessing.RESAMPLING_STRATEGIES),
                        help="Resampling strategies (default: all).")
    parser.add_argument("--model-params", default="{}", metavar="JSON",
                        help='JSON mapping model names to parameters; lists form a Cartesian grid; pipelines use classifier__PARAM keys.')
    parser.add_argument("--early-stopping-rounds", type=int, default=None,
                        help="XGBoost early-stopping patience (default: disabled; 0 disables). Per-model JSON overrides this.")
    parser.add_argument("--init", action="store_true", help="Create fixed stratified CSV splits and exit.")
    parser.add_argument("--export", action="store_true",
                        help="Save model, fitted preprocessing and JSON metadata bundles to results/models/; bypasses the feature cache.")
    parser.add_argument("--evaluate-model", type=Path, metavar="PATH",
                        help="Evaluate a saved bundle directory or metadata.json on dev (test with --final-evaluation); print results only.")
    parser.add_argument("--final-evaluation", action="store_true",
                        help="Explicitly evaluate selected models on reserved test.csv; no sampling.")
    parser.add_argument("--no-background", nargs="*", metavar="SPLIT",
                        help="Exclude Background from selected splits: train val dev test (spaces/commas accepted); flag alone means all loaded splits.")
    parser.add_argument("--cache-dir", type=Path, default=FeatureCache.CACHE_PATH,
                        help="Feature cache directory (default: cache/features).")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Recompute and replace the selected cached feature representations.")
    args = parser.parse_args(argv)
    if args.evaluate_model is not None:
        if (args.init or args.export or args.models is not None or args.features is not None
                or args.resampling is not None or args.sample_size is not None or args.multiclass
                or args.no_background is not None or args.model_params != "{}"
                or args.early_stopping_rounds is not None or args.rebuild_cache
                or args.cache_dir != FeatureCache.CACHE_PATH):
            parser.error("--evaluate-model accepts only --final-evaluation; model settings come from the saved bundle")
        return args
    if args.no_background is None:
        args.no_background = []
    elif not args.no_background or args.no_background == ["all"]:
        args.no_background = list(DataHandler.SPLIT_NAMES)
    else:
        selected = [part.strip() for value in args.no_background for part in value.split(",") if part.strip()]
        invalid = set(selected) - set(DataHandler.SPLIT_NAMES)
        if invalid or not selected:
            parser.error("--no-background accepts train, val, dev, test (or all)")
        args.no_background = list(dict.fromkeys(selected))
    if args.final_evaluation and (args.init or args.sample_size is not None or args.models is None):
        parser.error("--final-evaluation requires explicit --model(s) and cannot combine with --init, --sample-size")
    args.results = Evaluate.CLASSIFICATION_RESULTS_FILE_PATH if args.multiclass else Evaluate.RESULTS_FILE_PATH
    if args.multiclass and (args.models and "xgboost_scale_pos_weight" in args.models):
        parser.error("xgboost_scale_pos_weight is binary-only; use xgboost with --multiclass")
    if args.sample_size is not None and args.sample_size <= 0:
        parser.error("--sample-size must be positive")
    if args.early_stopping_rounds is not None and args.early_stopping_rounds < 0:
        parser.error("--early-stopping-rounds must be nonnegative")
    try:
        params = json.loads(args.model_params)
        if not isinstance(params, dict):
            raise ValueError("expected a JSON object")
        prototypes = Trainer.get_classifiers(target=pd.Series([0, 1]))
        for name, values in params.items():
            if name not in prototypes or not isinstance(values, dict):
                raise ValueError(f"invalid model or parameter object: {name}")
            if args.models is not None and name not in args.models:
                raise ValueError(f"parameters supplied for unselected model: {name}")
            unknown = values.keys() - prototypes[name][0].get_params().keys()
            if unknown:
                raise ValueError(f"unknown parameters for {name}: {sorted(unknown)}")
            for combination in Trainer.parameter_combinations(values):
                prototypes[name][0].set_params(**combination)
        for name in ("xgboost", "xgboost_scale_pos_weight"):
            if args.models is None or name in args.models:
                values = params.setdefault(name, {})
                if args.early_stopping_rounds is not None:
                    values.setdefault("early_stopping_rounds", args.early_stopping_rounds or None)
                patience = values.get("early_stopping_rounds")
                for candidate in patience if isinstance(patience, list) else [patience]:
                    if candidate is not None and (type(candidate) is not int or candidate <= 0):
                        raise ValueError(f"{name}.early_stopping_rounds candidates must be positive integers or null")
        args.model_params = params
    except (ValueError, TypeError) as error:
        parser.error(f"invalid --model-params: {error}")
    return args


def main(argv=None) -> None:
    """Outline of the tasks this pipeline performs"""
    args = parse_args(argv)
    if args.evaluate_model is not None:
        try:
            ModelArtifacts.evaluate_saved_model(args.evaluate_model, args.final_evaluation)
        except (OSError, ValueError, KeyError) as error:
            raise SystemExit(f"Cannot evaluate saved model: {error}") from error
        return
    if args.init:
        DataHandler.initialize_splits(DataHandler.DATA_FILE_PATH, DataHandler.SPLITS_PATH)
        return
    try:
        preprocessing = None
        if args.export:
            sample_feature_representations, preprocessing = ModelArtifacts.prepare_export_features(args)
        else:
            sample_feature_representations = FeatureCache.load_features(
                sample_size=args.sample_size, resampling=args.resampling, features=args.features,
                multiclass=args.multiclass, no_background=args.no_background,
                final_evaluation=args.final_evaluation, cache_dir=args.cache_dir,
                rebuild=args.rebuild_cache,
            )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    trained_models = blanket_search(
        sample_feature_representations, args.sample_size, args.models, args.model_params, args.no_background
    )
    Evaluate.evaluate_classifiers(
        trained_models, results_file_path=args.results,
        evaluation_split="final_test" if args.final_evaluation else "test",
        export_dir=Evaluate.MODELS_PATH if args.export else None,
        preprocessing=preprocessing,
    )


if __name__ == "__main__":
    main()
