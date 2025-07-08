from typing import Tuple, Iterable

import numpy as np
from sklearn.model_selection import train_test_split

from deepfake_detection.data.datasets import ListDataset
from deepfake_detection.data.dataset import Dataset


def split_dataset(dataset: Dataset,
                  test_size: float = None,
                  random_state: int = None,
                  shuffle: bool = True,
                  stratify: bool = True) -> Tuple[ListDataset, ListDataset]:
    """
    Splits a dataset into training and test sets.
    
    :param dataset: Dataset to split.
    :param test_size: Float specifying the fraction of instances in the test set.
                      The remainder will be in the train set.
    :param random_state: Integer specifying the random state for splitting the dataset.
    :param shuffle: Boolean specifying whether to shuffle the dataset before splitting.
                    If shuffle=False then stratify must be None.
    :param stratify: If set to True, data is split in a stratified fashion using the class labels.
    """
    # Create index array
    index_array = np.arange(0, len(dataset))

    # Get labels
    labels = [i.label for i in dataset] if stratify else None
    
    # Create split
    train_set, test_set = train_test_split(index_array,
                                           test_size=test_size,
                                           random_state=random_state,
                                           shuffle=shuffle,
                                           stratify=labels)

    # Index instances using splits
    train_instances = [instance for (instance, idx) in zip(dataset, index_array) if idx in train_set]
    test_instances = [instance for (instance, idx) in zip(dataset, index_array) if idx in test_set]

    return (ListDataset(name=dataset.name, instances=train_instances),
            ListDataset(name=dataset.name, instances=test_instances))