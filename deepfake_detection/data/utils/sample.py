from typing import Optional, Union

import numpy as np

from deepfake_detection.data.datasets import ListDataset
from deepfake_detection.data.dataset import Dataset, MapStyleDatasetMixin


def sample_n_per_class(
    dataset: Dataset,
    n: int,
    label: Optional[str],
    random_state: Optional[Union[int, None]] = None,
) -> MapStyleDatasetMixin:
    """
    Samples n instances from each source class in the given dataset.
    If n is larger than the number of instances in a class, the maximum number of instances will be returned.

    :param dataset: The dataset to sample from.
    :param n: The number of instances to sample from each class.
    :param label: The label in 'Annotation' to use for sampling.
    :param random_state: The random seed to use for sampling.
    :return: A sampled dataset.
    """
    # Raise error if n is not a positive integer
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    # Set random seed
    rng = np.random.default_rng(random_state)

    # Map labels to indices
    label_to_indices = {}
    for idx in range(len(dataset)):
        instance = dataset[idx]
        label_value = instance.annotation.get_label(label)

        if label_value not in label_to_indices:
            label_to_indices[label_value] = []
        label_to_indices[label_value].append(idx)

    # Select the subset of indices
    selected_indices = []
    for label_value, indices in label_to_indices.items():
        sample_size = min(n, len(indices))
        sampled = rng.choice(indices, size=sample_size, replace=False)
        selected_indices.extend(sampled)

    # Create the sampled dataset using only the selected items
    sampled_instances = [dataset[i] for i in selected_indices]

    return ListDataset(sampled_instances)
