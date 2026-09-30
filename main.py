"""
main.py is the entry point of the code, and allows you to control which parts of the code is being run.
"""

import DataHandler
import PreProcessing
import Visualization

def preprocess(data):
    """Preprocess the data, and preprocess sample for feature visualization"""
    preprocessed_data = PreProcessing.preprocess_data(data)
    sampled_data = PreProcessing.sample_data(
        preprocessed_data,
        Visualization.EXPLORATION_SAMPLE_SIZE,
    )

    pca_features = PreProcessing.get_pca_features(sampled_data)
    tsne_features = PreProcessing.get_tsne_features(sampled_data)
    umap_features = PreProcessing.get_umap_features(sampled_data)
    supervised_umap_features = PreProcessing.get_umap_features(sampled_data, supervised=True)
    return preprocessed_data, sampled_data, pca_features, tsne_features, umap_features, supervised_umap_features

def explore_data(data, preprocessed_data, sampled_data, pca_features, tsne_features, umap_features, supervised_umap_features):
    """Explore the data by visualizing and printing information"""
    Visualization.print_dataset_information(data)
    Visualization.create_exploration_visualizations(
        preprocessed_data,
        sampled_data,
        pca_features,
        tsne_features,
        umap_features,
        supervised_umap_features,
    )


def main() -> None:
    """Outline of the tasks this pipeline performs"""
    # read data
    data = DataHandler.read_data()
    # preprocess data
    preprocessed_data = preprocess(data)
    # visualize and explore data
    explore_data(data, *preprocessed_data)


if __name__ == "__main__":
    main()
