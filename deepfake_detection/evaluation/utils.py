import logging
from typing import Sequence, Tuple, Mapping

import numpy as np

from deepfake_detection.data.instance import Instance
from deepfake_detection.models.prediction import Prediction


LOGGER = logging.getLogger(__name__)

def get_labels(
        instances: Sequence[Instance],
        predictions: Sequence[Prediction],
) -> Sequence[str]:
    """
    Returns a set of unique labels among the ground-truth annotations
    in `instances` and the predicted labels in `predictions`
    (i.e. the keys in their `classification` attributes) combined.

    :param instances: Instances with ground-truth labels
    :param predictions: Model predictions with classification scores
    :return: Unique labels among ground-truth annotations and model predictions
    """
    a = {y.label for y in instances}
    b = {y for p in predictions for y in p.classification}
    print(a, b)
    return sorted(a | b)

def to_arrays(instances: Sequence[Instance],
              predictions: Sequence[Prediction],
              label: str,
              binary: bool = False) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    This function takes a series of predictions and instances and
    transforms them into binary arrays given a certain label.

    :param instances: Instances with ground-truth labels
    :param predictions: Model predictions with classification scores
    :param label: label that corresponds to `true` class.
    :param binary: If `True`, predictions are returned as one-hot encoded labels for the particular label
                   (true if the label has the highest confidence score).
                   If `False`, predictions are returned as the confidence score for the particular label.
    :return: Tuple of (y_true, y_pred) arrays.
    """
    if binary:
        y_pred = np.array([max(p.classification, key=p.classification.get) == label for p in predictions])
    else:
        y_pred = np.array([p.classification[label] for p in predictions])
    y_true = np.array([int(i.label == label) for i in instances])
    return y_true, y_pred

def map_fields(init_dict: Mapping[str, float], map_dict: Mapping[str, str]) -> Mapping[str, float]:
    """
    Changes keys of 'init_dict' using the mapping defined in `map_dict`.
    If a key in 'init_dict' is not present in 'map_dict', it will not be changed.
    If multiple values are mapped to the same key, the highest value will be stored.

    :param init_dict: a mapping of key-value pairs.
    :param map_dict: a mapping of key-key pairs.
    """
    res_dict = {}
    for k, v in init_dict.items():
        if k not in map_dict.keys():
            res_dict[k] = v
            continue
        k = str(map_dict[k])
        if k in res_dict.keys():
            if v <= res_dict[k]:
                continue
        res_dict[k] = v
    return res_dict

def apply_label_mapping(instances: Sequence[Instance],
                        predictions: Sequence[Prediction],
                        label_mapping: Mapping[str, str]) -> Tuple[Sequence[Instance], Sequence[Prediction]]:
    """
    This function applies a label mapping to a series of instances and a series of predictions and returns the same
    predictions and instances, but with updated labels according to the label mapping.

    :param instances: Instances with ground-truth labels
    :param predictions: Model predictions with classification scores
    :param label_mapping: Mapping from label to label.
    :return: Tuple of (instances, predictions) with updated labels.
    """
    # Update label mapping for predictions
    new_predictions = []
    for p in predictions:
        p.classification = map_fields(p.classification, label_mapping)
        new_predictions.append(p)

    # Update label mapping for instances
    new_instances = []
    for i in instances:
        i.label = label_mapping[i.label]
        new_instances.append(i)

    return new_instances, new_predictions