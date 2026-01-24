import logging
from typing import Sequence, Tuple, Mapping, Optional

import numpy as np

from deepfake_detection.data.instance import Instance
from deepfake_detection.models.prediction import Prediction


def get_labels(
    instances: Sequence[Instance],
    predictions: Sequence[Prediction],
    label_type: str = "authenticity",
) -> Sequence[str]:
    """
    Returns a set of unique labels among the ground-truth annotations
    in `instances` and the predicted labels in `predictions`
    (i.e. the keys in their `classification` attributes) combined.

    :param instances: Instances with ground-truth labels.
    :param predictions: Model predictions with classification scores.
    :param label_type: The label type in 'Annotation' to include.
    :return: Unique labels among ground-truth annotations and model predictions.
    """
    a = {y.annotation.get_label(label_type) for y in instances}
    b = {y for p in predictions for y in p.classification}
    if len(a & b) == 0:
        logging.warn("No common labels found in instances and predictions.")
    return sorted(a | b)


def to_arrays(
    instances: Sequence[Instance],
    predictions: Sequence[Prediction],
    label: str,
    label_type: str,
    threshold: Optional[float] = None,
    binary: Optional[bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function takes a series of predictions and instances and transforms them into arrays.

    :param instances: Instances with ground-truth labels
    :param predictions: Model predictions with classification scores.
    :param label: label that corresponds to positive class.
    :param label_type: The label type in 'Annotation' to use for y_true.
                       Should correspond with the labels in 'predictions'.
    :param threshold: An optional threshold to use for converting confidence scores to predicted labels.
    :param binary: If `True`, predictions are returned as one-hot encoded labels for the particular target class.
                   If `False`, predictions are returned as the confidence score for the particular label.
    :return: Tuple of (y_true, y_pred) arrays.
    """
    # If a threshold is set and binary labels should be returned, return one-hot labels higher than threshold
    if threshold and binary:
        y_pred = np.array(
            [int(p.classification[label] >= threshold) for p in predictions]
        )

    # Otherwise if no threshold is set and binary labels should be returned,
    # return one-hot labels of max confidence score
    elif binary:
        y_pred = np.array(
            [
                int(max(p.classification, key=p.classification.get) == label)
                for p in predictions
            ]
        )

    # Otherwise just return confidence score of selected label
    else:
        y_pred = np.array([p.classification[label] for p in predictions])

    # Get true labels of specified label type
    y_true = np.array(
        [int(label == i.annotation.get_label(label_type)) for i in instances]
    )

    # Raise warnings when certain labels are not found
    if y_pred.sum() == 0:
        logging.warn("No predictions for label %s.", label)
    if y_true.sum() == 0:
        logging.warn("No labels for label %s.", label)

    return y_true, y_pred


def map_fields(
    init_dict: Mapping[str, float], map_dict: Mapping[str, str]
) -> Mapping[str, float]:
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


def transform_prediction(
    prediction: Prediction, label_mapping: Mapping[str, str]
) -> Prediction:
    """
    Transforms the classification attribute of prediction by replacing the source labels with the target labels.
    The values of the target labels with multiple source labels will be summed.
    If a label is not in the mapping, the original label is kept.

    :param prediction: The prediction to transform.
    :param label_mapping: A mapping from source labels to target labels.
    """
    new_dict = {}
    for key, value in prediction.classification.items():
        new_key = label_mapping.get(key, key)  # map key or keep original if no mapping
        new_dict[new_key] = new_dict.get(new_key, 0) + value
    prediction.classification = new_dict
    return prediction
