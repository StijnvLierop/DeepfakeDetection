from typing import Union

import numpy as np

from deepfake_detection.data import Dataset
from deepfake_detection.data.datasets.filtered_dataset import FilteredDataset


def sample_n_per_class(dataset: Dataset, n: int, random_seed: Union[int, None] = None):
    """
    Samples n instances from each class in the given dataset.
    If n is larger than the number of instances in a class, the maximum number of instances will be returned.

    :param dataset: The dataset to sample from.
    :param n: The number of instances to sample from each class.
    :param random_seed: The random seed to use for sampling.
    :return: A filtered dataset containing the sampled instances.
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
        label = instance.label
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    # Sample n indices from each list
    indices = []
    for label in label_to_indices.keys():
        indices.extend(np.random.choice(label_to_indices[label], min(n, len(label_to_indices[label])), replace=False))

    # Return an iterable that contains n indices
    return FilteredDataset(dataset, indices)
