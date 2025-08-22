from typing import Sequence, Union

import numpy as np
import sklearn
from matplotlib import pyplot as plt

from deepfake_detection.data import Instance, Dataset
from deepfake_detection.evaluation.utils import get_labels, to_arrays, get_label_mapping
from deepfake_detection.models import Prediction


def confusion_matrix(instances: Union[Sequence[Instance], Dataset],
                     predictions: Sequence[Prediction],
                     label_type: str) -> np.ndarray:
    """
    Computes the confusion matrix for the annotated ``instances`` and corresponding ``predictions``.

    :param instances: Data with ground truth labels.
    :param predictions: The corresponding model predictions.
    :param label_type: The label type in 'Annotation' to use for computing the accuracy. The labels in the 'Prediction'
                       instances should correspond with the labels in this 'label_type' in Annotation.
                       - If 'predictions' contain labels for 'real', 'fake' and/or 'manipulated',
                         should be set to 'authenticity_label'.
                       - If 'predictions' contain labels for specific sources, should be set to 'source_label'.
                       - If 'predictions' contain binary labels, this parameter should be set to 'binary_label'.
    :return: Confusion matrix.
    """
    # Get labels on level of given 'label_type'
    labels = get_labels(instances, predictions, label_type)

    # Check if predictions and instances have the same length
    if len(predictions) != len(instances):
        raise ValueError("Predictions and instances must have the same length.")

    # Get y_pred and y_true
    y_pred = [max(p.classification, key=p.classification.get) for p in predictions]
    y_true = [i.annotation.get_label(label_type) for i in instances]

    # Calculate confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels)

    return cm


def plot_confusion_matrix(instances: Union[Sequence[Instance], Dataset],
                          predictions: Sequence[Prediction],
                          label_type: str) -> None:
    """
    Plots the confusion matrix for the annotated ``instances`` and corresponding ``predictions``.

    :param instances: Data with ground truth labels.
    :param predictions: The corresponding model predictions.
    :param label_type: The label type in 'Annotation' to use for computing the accuracy. The labels in the 'Prediction'
                       instances should correspond with the labels in this 'label_type' in Annotation.
                       - If 'predictions' contain labels for 'real', 'fake' and/or 'manipulated',
                         should be set to 'authenticity_label'.
                       - If 'predictions' contain labels for specific sources, should be set to 'source_label'.
                       - If 'predictions' contain binary labels, this parameter should be set to 'binary_label'.
    """

    # Get labels on level of given 'label_type'
    labels = get_labels(instances, predictions, label_type)

    # Calculate confusion matrix
    cm = confusion_matrix(instances, predictions, label_type)

    # Plot confusion matrix
    plt.figure(figsize=(20, 20))
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()