import os
from typing import List

import numpy as np
import pandas as pd

from deepfake_detection.data.datasets.dataset import Dataset


def write_predictions_to_file(results_dir: str,
                              predictions: List[np.ndarray],
                              dataset: Dataset,
                              model_name: str) -> str:
    """
    This function writes a set of predictions corresponding to a given dataset to a file.

    :param results_dir: The path to the directory where the predictions will be written to.
    :param predictions: The predictions to write to a file.
    :param dataset: The dataset corresponding to the predictions.
    :param model_name: The name of the model that made the predictions.
    :return: The path to the file where the predictions were written.
    """

    # Ensure that the length of prediction and dataset is the same
    if len(predictions) != len(dataset):
        raise ValueError("Predictions must have the same length as the dataset!")

    # Create dataframe
    df = pd.DataFrame([(sample.path, prediction, sample.label) for sample, prediction in zip(dataset, predictions)],
                      columns=['file', 'prediction', 'ground_truth'])

    # Write to file
    filename = os.path.join(results_dir, get_predictions_filename(dataset, model_name))
    df.to_csv(filename, index=False)

    return filename


def read_predictions_from_file(predictions_path: str) -> pd.DataFrame:
    """
    This function reads a set of predictions corresponding to a given dataset from a file.

    :param predictions_path: The path to the file where the predictions are stored.
    """
    return pd.read_csv(predictions_path, delimiter=',')


def get_predictions_filename(dataset_name: str, model_name: str) -> str:
    """
    This function returns a filename for a new predictions file.

    :param dataset_name: The dataset name corresponding to the predictions.
    :param model_name: The name of the model that made the predictions.
    """
    return f'predictions_{dataset_name}_{model_name}.csv'