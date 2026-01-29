from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.data.datasets import ListDataset


def split_dataset(
    dataset: Dataset,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    shuffle: Optional[bool] = True,
    stratify: Optional[bool] = False,
    label: Optional[str] = "source",
) -> Tuple[ListDataset, ListDataset]:
    """
    Splits a dataset into training and test sets.

    :param dataset: Dataset to split.
    :param test_size: Float specifying the fraction of instances in the test set.
                      The remainder will be in the train set.
    :param random_state: Integer specifying the random state for splitting the dataset.
    :param shuffle: Boolean specifying whether to shuffle the dataset before splitting.
                    If shuffle=False then stratify must be None.
    :param stratify: If set to True, data is split in a stratified fashion using the class labels type in 'label'.
    :param label: If stratify=True, this specifies the label to use for stratification.
    """
    # Create index array
    index_array = np.arange(0, len(dataset))

    # Get labels
    labels = [i.annotation.get_label(label) for i in dataset] if stratify else None

    # Create split
    train_set, test_set = train_test_split(
        index_array,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=labels,
    )

    # Index instances using splits
    train_instances = [
        instance for (instance, idx) in zip(dataset, index_array) if idx in train_set
    ]
    test_instances = [
        instance for (instance, idx) in zip(dataset, index_array) if idx in test_set
    ]

    # Set new dataset names
    dataset1_name = dataset.dataset_name + "_split1" if dataset.dataset_name else "split1"
    dataset2_name = dataset.dataset_name + "_split2" if dataset.dataset_name else "split2"

    return (
        ListDataset(dataset_name=dataset1_name, instances=train_instances),
        ListDataset(dataset_name=dataset2_name, instances=test_instances),
    )