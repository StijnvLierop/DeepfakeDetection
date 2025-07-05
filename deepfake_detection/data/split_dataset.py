from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from deepfake_detection.data.datasets import ListDataset
from deepfake_detection.data.dataset import Dataset


def split_dataset(dataset: Dataset,
                  test_size : float = None,
                  random_state : int = None) -> Tuple[ListDataset, ListDataset]:
    """
    Splits a dataset into training and test sets.

    :param dataset: Dataset to split.
    :param test_size: Float specifying the fraction of instances in the test set.
                      The remainder will be in the train set.
    :param random_state: Integer specifying the random state for splitting the dataset.
    """
    # Create index array
    index_array = np.arange(0, len(dataset))

    # Create split
    train_set, test_set = train_test_split(index_array,
                                           test_size=test_size,
                                           random_state=random_state)

    # Index instances using splits
    train_instances = [instance for (instance, idx) in zip(dataset, index_array) if idx in train_set]
    test_instances = [instance for (instance, idx) in zip(dataset, index_array) if idx in test_set]

    return (ListDataset(name=dataset.name, instances=train_instances),
            ListDataset(name=dataset.name, instances=test_instances))