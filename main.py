"""
main.py is the entry point of the code, and allows you to control which parts of the code is being run.
"""

import DataHandler
import PreProcessing
import Visualization
import Trainer
import Evaluate
from time import perf_counter

EXPLORATION_SAMPLE_SIZE = 10000

def preprocess(dataset):
    """Preprocess the data, and preprocess sample for feature visualization"""
    start_time = perf_counter()
    preprocessed_data = PreProcessing.preprocess_data(dataset)
    print(f"[timing] preprocessing clean numeric values: {perf_counter() - start_time:.2f}s")

    start_time = perf_counter()
    sample_dataset = PreProcessing.sample_data(
        preprocessed_data,
        EXPLORATION_SAMPLE_SIZE,
    )
    print(f"[timing] preprocessing sample data: {perf_counter() - start_time:.2f}s")

    resampled_datasets = {
        "no_resampling": sample_dataset,
    }

    start_time = perf_counter()
    resampled_datasets["undersample_majority"] = PreProcessing.majority_class_undersampling(sample_dataset)
    print(f"[timing] preprocessing undersample majority: {perf_counter() - start_time:.2f}s")

    start_time = perf_counter()
    resampled_datasets["oversample_minority"] = PreProcessing.minority_class_oversampling(sample_dataset)
    print(f"[timing] preprocessing oversample minority: {perf_counter() - start_time:.2f}s")

    feature_representations = {}
    for resampling_strategy, resampled_dataset in resampled_datasets.items():
        print(f"[timing] building feature representations for {resampling_strategy}")
        feature_representations[resampling_strategy] = PreProcessing.get_all_feature_representations(
            resampled_dataset
        )

    return (preprocessed_data,
            sample_dataset,
            feature_representations)

def explore_data(data, preprocessed_data, sampled_data):
    """Explore the data by visualizing and printing information"""
    Visualization.print_dataset_information(data)
    Visualization.create_exploration_visualizations(
        preprocessed_data,
        sampled_data,
    )

def blanket_search(feature_representations_by_resampling: dict):
    trained_models = []
    for resampling_strategy, feature_representations in feature_representations_by_resampling.items():
        trained_models.extend(
            Trainer.compare_feature_representations_and_classifiers(
                feature_representations,
                EXPLORATION_SAMPLE_SIZE,
                resampling_strategy=resampling_strategy,
            )
        )
    return trained_models

def evaluate(trained_models):
    return Evaluate.evaluate_classifiers(trained_models)


def main() -> None:
    """Outline of the tasks this pipeline performs"""
    # read data
    full_data = DataHandler.read_data()
    dataset = DataHandler.get_stratified_split(full_data) # we will not use val or test for now

    # preprocess data
    preprocessed_data, sample_dataset, sample_feature_representations = preprocess(dataset)

    # visualize and explore data
    explore_data(dataset, preprocessed_data, sample_dataset)
    # train combining feature representations and classifiers
    trained_models = blanket_search(sample_feature_representations)
    # evaluate trained models
    evaluate(trained_models)


if __name__ == "__main__":
    main()
