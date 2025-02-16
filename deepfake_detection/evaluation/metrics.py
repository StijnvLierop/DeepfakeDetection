from typing import Sequence

import numpy as np
import sklearn

from deepfake_detection.data.datasets.instance import Instance
from deepfake_detection.models.prediction import Prediction


def accuracy(
        instances: Sequence[Instance],
        predictions: Sequence[Sequence[Prediction]],
        label_type: str = 'authenticity_label',
        positive_label: str = None
) -> float:
    """
    Computes the accuracy for the annotated ``instances`` and corresponding
    ``predictions``.

    :param instances: Data with ground truth labels.
    :param predictions: The corresponding model predictions.
    :param positive_label: The label to use for the positive class.
    :param label_type: The label to use: 'authenticity_label' or 'class_label'.
    :return: Accuracy
    """
    # Get true labels
    if label_type == 'authenticity_label':
        y_true = []
        y_pred = []
        for p, i in zip(predictions, instances):
            label = int(i.authenticity_label == positive_label)
            pred = int(max(p.classification, key=p.classification.get) == i.authenticity_label)
            y_true.append(label)
            y_pred.append(pred)

    elif label_type == 'class_label':
        y_true = np.array([1 if i.class_label == positive_label else 0 for i in instances])
        y_pred = [1 if max(p.classification, key=p.classification.get) == positive_label else 0 for p in predictions]
    else:
        raise ValueError('label_type must be "authenticity_label" or "class_label"')

    if sum(y_true) == 0:
        print("Warning: No instances with positive label found. Was the right label provided?")

    # Calculate accuracy
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)

    return acc