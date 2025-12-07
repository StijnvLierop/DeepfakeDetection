import logging
from typing import Sequence, Tuple, Mapping, Optional, Union, Iterable

import numpy as np

from deepfake_detection.data.instance import Instance
from deepfake_detection.models.prediction import Prediction


def get_labels(
        instances: Sequence[Instance],
        predictions: Sequence[Prediction],
        label_type: str = 'authenticity_label'
) -> Sequence[str]:
    """
    Returns a set of unique labels among the ground-truth annotations
    in `instances` and the predicted labels in `predictions`
    (i.e. the keys in their `classification` attributes) combined.

    :param instances: Instances with ground-truth labels.
    :param predictions: Model predictions with classification scores.
    :param label_type: The label type in 'Annotation' to include. Can be 'source_label', 'authenticity_label'
                       or 'binary_label'.
    :return: Unique labels among ground-truth annotations and model predictions.
    """
    a = {y.annotation.get_label(label_type) for y in instances}
    b = {y for p in predictions for y in p.classification}
    if len(a & b) == 0:
        logging.warn("No common labels found in instances and predictions.")
    return sorted(a | b)


def to_arrays(instances: Sequence[Instance],
              predictions: Sequence[Prediction],
              label: str,
              label_type: str,
              binary: Optional[bool] = False) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    This function takes a series of predictions and instances and
    transforms them into binary arrays given a certain label.

    :param instances: Instances with ground-truth labels
    :param predictions: Model predictions with classification scores
    :param label: label that corresponds to `true` class.
    :param label_type: The label type in 'Annotation' to use for computing the accuracy. Can be 'source_label',
                       'authenticity_label' or 'binary_label'. Should correspond with the labels in 'predictions'.
    :param binary: If `True`, predictions are returned as one-hot encoded labels for the particular label
                   (true if the label has the highest confidence score).
                   If `False`, predictions are returned as the confidence score for the particular label.
    :return: Tuple of (y_true, y_pred) arrays.
    """
    if binary:
        y_pred = np.array([max(p.classification, key=p.classification.get) == label for p in predictions])
    else:
        y_pred = np.array([p.classification[label] for p in predictions])
    y_true = np.array([int(label == i.annotation.get_label(label_type)) for i in instances])
    if y_pred.sum() == 0:
        logging.warn("No predictions for label %s.", label)
    if y_true.sum() == 0:
        logging.warn("No labels for label %s.", label)
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


def find_label_type_corresponding_with_label(instances: Iterable[Instance], label: Union[str, int]) -> str:
    """
    Determines the type of label corresponding to a given label within a list of instances.

    :param instances: A list where each element is expected to have an 'annotation' attribute containing the label types.
    :param label: The label whose type is being determined.
    :return: The type of the label ('authenticity_label', 'source_label', 'binary_label') corresponding to
             the provided label. Raises a ValueError if the label is not found in the instances.
    """
    # Loop over instances
    for i in instances:

        # Check in which label type label occurs, return label type if found
        if i.annotation.authenticity_label == label:
            return 'authenticity_label'
        elif i.annotation.source_label == label:
            return 'source_label'
        elif i.annotation.binary_label == label:
            return 'binary_label'

    raise ValueError("Label not found in instances.")


def get_label_mapping(instances: Iterable[Instance], source_label: str, target_label: str) -> Mapping[str, str]:
    """
    Returns a mapping from source labels to target labels based on the labels of the provided instances.
    If source labels occur multiple times, a random target label will be chosen for each source label.

    :param instances: A list of instances.
    :param source_label: The label type to use as source for the mapping.
                         Can be one of 'authenticity_label', 'source_label' or 'binary_label'.
    :param target_label: The label type to use as source for the mapping.
                         Can be one of 'authenticity_label', 'source_label' or 'binary_label'.
    :return: A mapping from source labels to target labels.
    """
    # Ensure source and target label parameters are valid
    if source_label not in ['authenticity_label', 'source_label', 'binary_label']:
        raise ValueError("Invalid source label. Must be one of 'authenticity_label', 'source_label' or 'binary_label'.")
    if target_label not in ['authenticity_label', 'source_label', 'binary_label']:
        raise ValueError("Invalid target label. Must be one of 'authenticity_label', 'source_label' or 'binary_label'.")

    # Create mapping
    mapping = {}

    # Loop over instances
    for i in instances:
        mapping[i.annotation.get_label(source_label)] = i.annotation.get_label(target_label)

    return mapping


def transform_prediction(prediction: Prediction, label_mapping: Mapping[str, str]) -> Prediction:
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