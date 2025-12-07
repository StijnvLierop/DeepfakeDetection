import warnings
from typing import Sequence, Mapping, Union, Optional

import numpy as np
import sklearn

from deepfake_detection.data import Dataset
from deepfake_detection.data.instance import Instance
from deepfake_detection.evaluation.utils import get_labels, to_arrays, find_label_type_corresponding_with_label
from deepfake_detection.models.prediction import Prediction


def accuracy(instances: Union[Sequence[Instance], Dataset],
             predictions: Sequence[Prediction],
             label_type: str,
             label: Optional[str] = None) -> float:
    """
    Computes the accuracy for the annotated ``instances`` and corresponding ``predictions``.

    :param instances: Data with ground truth labels.
    :param predictions: The corresponding model predictions.
    :param label_type: The label type in 'Annotation' to use for computing the accuracy. The labels in the 'Prediction'
                       instances should correspond with the labels in this 'label_type' in Annotation.
                       - If 'predictions' contain labels for 'real', 'fake' and/or 'manipulated',
                         should be set to 'authenticity_label'.
                       - If 'predictions' contain labels for specific sources, should be set to 'source_label'.
                       - If 'predictions' contain binary labels, this parameter should be set to 'binary_label'.
    :param label: The label to compute the accuracy for. If omitted, the average score across all labels in the
                  'label_type' category is computed. If set to a label from a different category than 'label_type',
                   accuracy for the given label is computed based on the predictions for the label in 'label_type'.
    :return: Accuracy.
    """
    # Get labels on level of given 'label_type'
    labels = get_labels(instances, predictions, label_type)

    # Check if predictions and instances have the same length
    if len(predictions) != len(instances):
        raise ValueError("Predictions and instances must have the same length.")

    # If label provided, calculate accuracy for that label
    if label:

        # If label is not in labels of 'label_type'
        if label not in labels:

            # Find label_type label corresponding with label
            label_type_of_label = find_label_type_corresponding_with_label(instances, label)

            # Filter instances and predictions on instances which have label in annotation
            instances, predictions = zip(*[(i, p) for (i, p) in zip(instances, predictions)
                                           if i.annotation.get_label(label_type_of_label) == label])

            # Set label to a label in label_type
            label = instances[0].annotation.get_label(label_type)

        # Make classification based on label and
        y_true, y_pred = to_arrays(instances, predictions, label, label_type, binary=True)
        return sklearn.metrics.accuracy_score(y_true, y_pred)

    # Otherwise, get labels and return average score for all labels in 'label_type'
    scores = []
    for label in labels:
        y_true, y_pred = to_arrays(instances, predictions, label, label_type, binary=True)
        scores.append(sklearn.metrics.accuracy_score(y_true, y_pred))

    return float(np.mean(scores))


def roc_auc(instances: Union[Sequence[Instance], Dataset],
            predictions: Sequence[Prediction],
            label_type: str,
            label: str = None) -> float:
    """
    Computes the ROC-AUC score for the annotated ``instances`` and corresponding ``predictions``.

    :param instances: Data with ground truth labels.
    :param predictions: The corresponding model predictions.
    :param label_type: The label type in 'Annotation' to use for computing the roc_auc. The labels in the 'Prediction'
                       instances should correspond with the labels in this 'label_type' in Annotation.
                       - If 'predictions' contain labels for 'real', 'fake' and/or 'manipulated',
                         should be set to 'authenticity_label'.
                       - If 'predictions' contain labels for specific sources, should be set to 'source_label'.
                       - If 'predictions' contain binary labels, this parameter should be set to 'binary_label'.
    :param label: The label to compute the roc-auc for. If omitted, the average score across all labels in the
                  'label_type' category is computed. The label should occur in 'label_type'.
    :return: ROC-AUC.
    """
    # Get labels
    labels = get_labels(instances, predictions, label_type)

    # Check if predictions and instances have the same length
    if len(predictions) != len(instances):
        raise ValueError("Predictions and instances must have the same length.")

    # If label provided, calculate accuracy for that label
    if label:

        # If label is not in labels of 'label_type'
        if label not in labels:
            raise ValueError(f"Label {label} not found in instances.")

        y_true, y_pred = to_arrays(instances, predictions, label, label_type)
        return sklearn.metrics.roc_auc_score(y_true, y_pred)

    # Otherwise, get labels and return average score for all labels
    scores = []
    for label in labels:
        y_true, y_pred = to_arrays(instances, predictions, label, label_type)
        scores.append(sklearn.metrics.roc_auc_score(y_true, y_pred))

    return float(np.mean(scores))