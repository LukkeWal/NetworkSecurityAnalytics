"""
main.py is the entry point of the code, and allows you to control which parts of the code is being run.
"""

import argparse
import json
from pathlib import Path

import DataHandler
import PreProcessing
import Visualization
import Trainer
import Evaluate
import pandas as pd
from time import perf_counter

EXPLORATION_SAMPLE_SIZE = 10000

def preprocess(dataset, sample_size=EXPLORATION_SAMPLE_SIZE, resampling=None, features=None):
    """Preprocess the data, and preprocess sample for feature visualization"""
    start_time = perf_counter()
    preprocessed_data = PreProcessing.preprocess_data(dataset)
    print(f"[timing] preprocessing clean numeric values: {perf_counter() - start_time:.2f}s")

    start_time = perf_counter()
    sample_dataset = PreProcessing.sample_data(
        preprocessed_data,
        sample_size,
    )
    print(f"[timing] preprocessing sample data: {perf_counter() - start_time:.2f}s")

    strategies = resampling or ["no_resampling", "undersample_majority", "oversample_minority"]
    builders = {
        "no_resampling": lambda data: data,
        "undersample_majority": PreProcessing.majority_class_undersampling,
        "oversample_minority": PreProcessing.minority_class_oversampling,
    }
    resampled_datasets = {name: builders[name](sample_dataset) for name in strategies}

    feature_representations = {}
    for resampling_strategy, resampled_dataset in resampled_datasets.items():
        print(f"[timing] building feature representations for {resampling_strategy}")
        feature_representations[resampling_strategy] = PreProcessing.get_all_feature_representations(
            resampled_dataset, selected=features
        )

    return (preprocessed_data,
            sample_dataset,
            feature_representations)

def explore_data(data, preprocessed_data, sampled_data, figures_path=Visualization.FIGURES_PATH):
    """Explore the data by visualizing and printing information"""
    Visualization.print_dataset_information(data)
    Visualization.create_exploration_visualizations(
        preprocessed_data,
        sampled_data,
        figures_path=figures_path,
    )

def blanket_search(feature_representations_by_resampling: dict, sample_size=EXPLORATION_SAMPLE_SIZE, models=None, model_params=None):
    trained_models = []
    for resampling_strategy, feature_representations in feature_representations_by_resampling.items():
        trained_models.extend(
            Trainer.compare_feature_representations_and_classifiers(
                feature_representations,
                sample_size,
                resampling_strategy=resampling_strategy,
                model_names=models,
                model_params=model_params,
            )
        )
    return trained_models

def evaluate(trained_models):
    return Evaluate.evaluate_classifiers(trained_models)


FEATURE_NAMES = ["raw_numeric", "pca", "pca_95"] + [
    f"{kind}_{count}" for count in (5, 10, 20) for kind in ("umap", "umap_supervised")
]
MODEL_NAMES = list(Trainer.get_classifiers()) + ["xgboost_scale_pos_weight"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate selected network traffic classifiers.")
    parser.add_argument("--models", "--model", nargs="+", choices=MODEL_NAMES,
                        help="Models to train. With visualization flags alone, no models are trained; with no flags, runs all models.")
    parser.add_argument("--features", "--feature", nargs="+", choices=FEATURE_NAMES,
                        help="Feature representations (default: all).")
    parser.add_argument("--sample-size", "--exploration-sample-size", type=int,
                        default=EXPLORATION_SAMPLE_SIZE, help="Total sample size across train/val/test (default: 10000).")
    parser.add_argument("--resampling", nargs="+",
                        choices=["no_resampling", "undersample_majority", "oversample_minority"],
                        help="Resampling strategies (default: all).")
    parser.add_argument("--model-params", default="{}", metavar="JSON",
                        help='JSON mapping model names to parameters; pipelines use classifier__PARAM keys.')
    parser.add_argument("--early-stopping-rounds", type=int, default=20,
                        help="XGBoost validation-loss patience (default: 20; 0 disables). Per-model JSON overrides this.")
    parser.add_argument("--data", type=Path, help="Input CSV (default: project dataset).")
    parser.add_argument("--results", type=Path, default=Evaluate.RESULTS_FILE_PATH, help="Results CSV path.")
    parser.add_argument("--figures", type=Path, default=Visualization.FIGURES_PATH, help="Plot output directory.")
    parser.add_argument("--exploration", action="store_true", help="Enable dataset exploration reports and plots (default: disabled).")
    parser.add_argument("--learning-curves", action="store_true", help="Enable learning curve plots (losses are always recorded).")
    args = parser.parse_args(argv)
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")
    if args.early_stopping_rounds < 0:
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
            prototypes[name][0].set_params(**values)
        for name in ("xgboost", "xgboost_scale_pos_weight"):
            if args.models is None or name in args.models:
                values = params.setdefault(name, {})
                values.setdefault("early_stopping_rounds", args.early_stopping_rounds or None)
                patience = values["early_stopping_rounds"]
                if patience is not None and (type(patience) is not int or patience <= 0):
                    raise ValueError(f"{name}.early_stopping_rounds must be a positive integer or null")
        args.model_params = params
    except (ValueError, TypeError) as error:
        parser.error(f"invalid --model-params: {error}")
    return args


def main(argv=None) -> None:
    """Outline of the tasks this pipeline performs"""
    args = parse_args(argv)
    visualization_only = args.models is None and (args.exploration or args.learning_curves)
    if visualization_only and not args.exploration:
        Visualization.create_learning_curve_visualizations(args.results, args.figures)
        return
    # read data
    full_data = DataHandler.read_data() if args.data is None else pd.read_csv(args.data)
    dataset = DataHandler.get_stratified_split(full_data)

    # preprocess data
    preprocessed_data, sample_dataset, sample_feature_representations = preprocess(
        dataset, args.sample_size, args.resampling, args.features
    )

    # visualize and explore data
    if args.exploration:
        args.figures.mkdir(parents=True, exist_ok=True)
        explore_data(dataset, preprocessed_data, sample_dataset, args.figures)
    # train combining feature representations and classifiers
    if not visualization_only:
        trained_models = blanket_search(
            sample_feature_representations, args.sample_size, args.models, args.model_params
        )
        # evaluate trained models
        Evaluate.evaluate_classifiers(trained_models, results_file_path=args.results)
    if args.learning_curves:
        Visualization.create_learning_curve_visualizations(args.results, args.figures)


if __name__ == "__main__":
    main()
