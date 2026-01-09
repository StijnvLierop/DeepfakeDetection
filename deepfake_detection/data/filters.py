from typing import Optional, Union, List

import numpy as np

from deepfake_detection.data.dataset import MapStyleDatasetMixin


def sample_n_per_class_filter(
    dataset: MapStyleDatasetMixin,
    n: int,
    label_type: Optional[str] = "source_label",
    random_seed: Optional[Union[int, None]] = None,
) -> List[int]:
    """
    Samples n instances from each source class in the given dataset.
    If n is larger than the number of instances in a class, the maximum number of instances will be returned.

    :param dataset: The dataset to sample from.
    :param n: The number of instances to sample from each class.
    :param label_type: The label type in 'Annotation' to use for sampling.
                       Can be 'source_label' or 'authenticity_label'. Defaults to 'source_label'.
    :param random_seed: The random seed to use for sampling.
    :return: A list of indices of sampled instances.
    """

    # Return ValueError if n is invalid
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    # Set numpy random state
    if random_seed:
        np.random.seed(random_seed)

    # Create a label-to-index mapping
    label_to_indices = {}
    for idx, instance in enumerate(dataset):
        label = instance.annotation.get_label(label_type)
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    # Sample n indices from each list
    indices = []
    for label in label_to_indices.keys():
        indices.extend(
            np.random.choice(
                label_to_indices[label],
                min(n, len(label_to_indices[label])),
                replace=False,
            )
        )

    # Return sampled indices
    return indices