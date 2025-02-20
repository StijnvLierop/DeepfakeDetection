import warnings
from typing import Sequence, Mapping

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from deepfake_detection.data.datasets.instance import Instance
from deepfake_detection.evaluation.utils import get_labels, to_arrays, apply_label_mapping
from deepfake_detection.models.prediction import Prediction


def accuracy(instances: Sequence[Instance],
             predictions: Sequence[Prediction],
             label: str = None,
             label_mapping: Mapping[str, str] = None) -> float:
    """
    Computes the accuracy for the annotated ``instances`` and corresponding ``predictions``.

    :param instances: Data with ground truth labels.
    :param predictions: The corresponding model predictions.
    :param label: The label to compute the metric for. If omitted, the average score across all labels is
                  computed instead. If a label mapping is provided this label should also occur in the label mapping.
    :param label_mapping: A mapping from current to new label. This can be used to bin certain labels
                          in categories. If omitted, the given labels will be used. If multiple predicted labels
                          are mapped to the same new label, the highest confidence score value will be used to
                          determine the predicted class.
    :return: accuracy.
    """
    # If label mapping, transform labels
    if label_mapping:
        instances, predictions = apply_label_mapping(instances, predictions, label_mapping)

    # Get labels
    labels = get_labels(instances, predictions)

    # Check if provided label in labels
    if label and label not in labels:
        warnings.warn("Provided label does not exist in labels.")

    # Check if predictions and instances have the same length
    if len(predictions) != len(instances):
        raise ValueError("Predictions and instances must have the same length.")

    # If label provided, calculate metric for that label
    if label:
        y_true, y_pred = to_arrays(instances, predictions, label, binary=True)
        return accuracy_score(y_true, y_pred)

    # Otherwise, get labels and return average score for all labels
    scores = []
    for label in labels:
        y_true, y_pred = to_arrays(instances, predictions, label, binary=True)
        scores.append(accuracy_score(y_true, y_pred))

    return float(np.mean(scores))

def roc_auc(instances: Sequence[Instance],
            predictions: Sequence[Prediction],
            label: str = None,
            label_mapping: Mapping[str, str] = None) -> float:
    """
    Computes the ROC-AUC score for the annotated ``instances`` and corresponding ``predictions``.

    :param instances: Data with ground truth labels.
    :param predictions: The corresponding model predictions.
    :param label: The label to compute the metric for. If omitted, the average score across all labels is
                  computed instead. If a label mapping is provided this label should also occur in the label mapping.
    :param label_mapping: A mapping from current to new label. This can be used to bin certain labels
                          in categories. If omitted, the given labels will be used. If multiple predicted labels
                          are mapped to the same new label, the highest confidence score value will be used to
                          determine the predicted class.
    :return: ROC-AUC.
    """
    # If label mapping, transform labels
    if label_mapping:
        instances, predictions = apply_label_mapping(instances, predictions, label_mapping)

    # Get labels
    labels = get_labels(instances, predictions)

    # Check if provided label in labels
    if label and label not in labels:
        warnings.warn("Provided label does not exist in labels.")

    # Check if predictions and instances have the same length
    if len(predictions) != len(instances):
        raise ValueError("Predictions and instances must have the same length.")

    # If label provided, calculate metric for that label
    if label:
        y_true, y_pred = to_arrays(instances, predictions, label)
        return roc_auc_score(y_true, y_pred)

    # Otherwise, get labels and return average score for all labels
    scores = []
    for label in labels:
        y_true, y_pred = to_arrays(instances, predictions, label)
        scores.append(roc_auc_score(y_true, y_pred))

    return float(np.mean(scores))