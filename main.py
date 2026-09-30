"""
main.py is the entry point of the code, and allows you to control which parts of the code is being run.
"""

import DataHandler
import PreProcessing
import Visualization
import Trainer
import Evaluate

EXPLORATION_SAMPLE_SIZE = 2000

def preprocess(dataset):
    """Preprocess the data, and preprocess sample for feature visualization"""
    preprocessed_data = PreProcessing.preprocess_data(dataset)
    sample_dataset = PreProcessing.sample_data(
        preprocessed_data,
        EXPLORATION_SAMPLE_SIZE,
    )
    feature_representations = PreProcessing.get_all_feature_representations(sample_dataset)
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

def blanket_search(feature_representations: dict):
    trained_models = Trainer.compare_feature_representations_and_classifiers(
        feature_representations,
        EXPLORATION_SAMPLE_SIZE,
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
