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

EXPLORATION_SAMPLE_SIZE = None

def preprocess(dataset, resampling=None, features=None):
    """Preprocess the data, and preprocess sample for feature visualization"""
    start_time = perf_counter()
    preprocessed_data = PreProcessing.preprocess_data(dataset)
    print(f"[timing] preprocessing clean numeric values: {perf_counter() - start_time:.2f}s")

    strategies = resampling or ["no_resampling", "undersample_majority", "oversample_minority"]
    builders = {
        "no_resampling": lambda data: data,
        "undersample_majority": PreProcessing.majority_class_undersampling,
        "oversample_minority": PreProcessing.minority_class_oversampling,
    }
    resampled_datasets = {name: builders[name](preprocessed_data) for name in strategies}

    feature_representations = {}
    for resampling_strategy, resampled_dataset in resampled_datasets.items():
        print(f"[timing] building feature representations for {resampling_strategy}")
        feature_representations[resampling_strategy] = PreProcessing.get_all_feature_representations(
            resampled_dataset, selected=features
        )

    return preprocessed_data, feature_representations

def explore_data(data, preprocessed_data, sampled_data, figures_path=Visualization.FIGURES_PATH):
    """Explore the data by visualizing and printing information"""
    Visualization.print_dataset_information(data)
    Visualization.create_exploration_visualizations(
        preprocessed_data,
        sampled_data,
        figures_path=figures_path,
    )

def blanket_search(feature_representations_by_resampling: dict, sample_size=EXPLORATION_SAMPLE_SIZE, models=None, model_params=None, no_background=None):
    trained_models = []
    for resampling_strategy, feature_representations in feature_representations_by_resampling.items():
        trained_models.extend(
            Trainer.compare_feature_representations_and_classifiers(
                feature_representations,
                sample_size,
                resampling_strategy=resampling_strategy,
                model_names=models,
                model_params=model_params,
                no_background=no_background,
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
                        default=EXPLORATION_SAMPLE_SIZE, help="Sample rows from train.csv and split 60/10/20/10 (last 10% withheld); omit to use fixed development splits.")
    parser.add_argument("--resampling", nargs="+",
                        choices=["no_resampling", "undersample_majority", "oversample_minority"],
                        help="Resampling strategies (default: all).")
    parser.add_argument("--model-params", default="{}", metavar="JSON",
                        help='JSON mapping model names to parameters; pipelines use classifier__PARAM keys.')
    parser.add_argument("--early-stopping-rounds", type=int, default=20,
                        help="XGBoost validation-loss patience (default: 20; 0 disables). Per-model JSON overrides this.")
    parser.add_argument("--init", action="store_true", help="Create fixed stratified CSV splits and exit.")
    parser.add_argument("--final-evaluation", action="store_true",
                        help="Explicitly evaluate selected models on reserved test.csv; no sampling or exploration.")
    parser.add_argument("--no-background", nargs="*", metavar="SPLIT",
                        help="Exclude Background from selected splits: train val test final_test (spaces/commas accepted); flag alone means all loaded splits.")
    parser.add_argument("--data", type=Path, help="Source CSV for --init only (default: project dataset).")
    parser.add_argument("--splits-dir", type=Path, default=DataHandler.SPLITS_PATH,
                        help="Fixed train/val/development/test CSV directory (default: data/splits).")
    parser.add_argument("--results", type=Path, help="Results CSV (default: evaluation_results.csv, or final_evaluation_results.csv with --final-evaluation).")
    parser.add_argument("--figures", type=Path, default=Visualization.FIGURES_PATH, help="Plot output directory.")
    parser.add_argument("--exploration", action="store_true", help="Enable dataset exploration reports and plots (default: disabled).")
    parser.add_argument("--learning-curves", action="store_true", help="Enable learning curve plots (losses are always recorded).")
    parser.add_argument("--confusion-matrices", action="store_true",
                        help="Plot default/tuned test confusion matrices (without models, reads existing CSV only).")
    args = parser.parse_args(argv)
    if args.no_background is None:
        args.no_background = []
    elif not args.no_background or args.no_background == ["all"]:
        args.no_background = list(DataHandler.SPLIT_NAMES)
    else:
        selected = [part.strip() for value in args.no_background for part in value.split(",") if part.strip()]
        invalid = set(selected) - set(DataHandler.SPLIT_NAMES)
        if invalid or not selected:
            parser.error("--no-background accepts train, val, test, final_test (or all)")
        args.no_background = list(dict.fromkeys(selected))
    if args.final_evaluation and (args.init or args.sample_size is not None or args.exploration or args.models is None):
        parser.error("--final-evaluation requires explicit --model(s) and cannot combine with --init, --sample-size, or --exploration")
    if args.results is None:
        args.results = Path("final_evaluation_results.csv") if args.final_evaluation else Evaluate.RESULTS_FILE_PATH
    if args.data is not None and not args.init:
        parser.error("--data is only used with --init; use --splits-dir to select initialized data")
    if args.sample_size is not None and args.sample_size <= 0:
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
    if args.init:
        DataHandler.initialize_splits(args.data or DataHandler.DATA_FILE_PATH, args.splits_dir)
        return
    visualization_only = args.models is None and (args.exploration or args.learning_curves or args.confusion_matrices)
    if visualization_only and not args.exploration:
        if args.learning_curves:
            Visualization.create_learning_curve_visualizations(args.results, args.figures)
        if args.confusion_matrices:
            Visualization.create_confusion_matrix_visualizations(args.results, args.figures)
        return
    # read data
    try:
        dataset = DataHandler.load_training_sample(
            args.sample_size, args.splits_dir, no_background=args.no_background,
            final_evaluation=args.final_evaluation,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    # preprocess data
    preprocessed_data, sample_feature_representations = preprocess(
        dataset, args.resampling, args.features
    )

    # visualize and explore data
    if args.exploration:
        args.figures.mkdir(parents=True, exist_ok=True)
        explore_data(dataset, preprocessed_data, preprocessed_data, args.figures)
    # train combining feature representations and classifiers
    if not visualization_only:
        trained_models = blanket_search(
            sample_feature_representations, args.sample_size, args.models, args.model_params, args.no_background
        )
        # evaluate trained models
        Evaluate.evaluate_classifiers(
            trained_models, results_file_path=args.results,
            evaluation_split="final_test" if args.final_evaluation else "test",
        )
    if args.learning_curves:
        Visualization.create_learning_curve_visualizations(args.results, args.figures)
    if args.confusion_matrices:
        Visualization.create_confusion_matrix_visualizations(args.results, args.figures)


if __name__ == "__main__":
    main()
