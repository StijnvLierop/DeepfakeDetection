import csv
import os
from typing import List

import numpy as np
import pandas as pd

from deepfake_detection.data.datasets.dataset import Dataset


def write_predictions_to_file(results_path: str,
                              predictions: List[np.ndarray],
                              dataset: Dataset,
                              model_name: str):
    """
    This function writes a set of predictions corresponding to a given dataset to a file.
    :param results_path:
    :param predictions:
    :param dataset:
    :param model_name:
    :return:
    """

    # Ensure that the length of prediction and dataset is the same
    assert len(predictions) == len(list(dataset))

    # Write to file
    fieldnames = ['file', 'prediction', 'ground_truth']
    filename = os.path.join(results_path, get_predictions_filename(dataset, model_name))
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sample, prediction in zip(dataset, predictions):
            writer.writerow({'file': sample.path, 'prediction': prediction[0], 'ground_truth': sample.label})

def read_predictions_from_file(predictions_path: str) -> pd.DataFrame:
    return pd.read_csv(predictions_path, delimiter=',')


def get_predictions_filename(dataset: Dataset, model_name: str) -> str:
    return f'predictions_{dataset.name}_{model_name}.csv'