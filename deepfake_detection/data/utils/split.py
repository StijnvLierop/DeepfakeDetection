from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from deepfake_detection.data.dataset import MapStyleDatasetMixin
from deepfake_detection.data.datasets import SubsetDataset


def split_dataset(
    dataset: MapStyleDatasetMixin,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    shuffle: Optional[bool] = True,
    stratify: Optional[bool] = False,
    label: Optional[str] = "source",
) -> Tuple[SubsetDataset, SubsetDataset]:
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

    # Get labels (only needed if stratify=True)
    if stratify:
        labels = [i.annotation.get_label(label) for i in dataset]
    else:
        labels = None

    # Create split
    train_set, test_set = train_test_split(
        index_array,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=labels,
    )

    # Set new dataset names
    dataset1_name = dataset.dataset_name + "_split1" if dataset.dataset_name else "split1"
    dataset2_name = dataset.dataset_name + "_split2" if dataset.dataset_name else "split2"

    return (
        SubsetDataset(dataset=dataset, dataset_name=dataset1_name, indices=train_set),
        SubsetDataset(dataset=dataset, dataset_name=dataset2_name, indices=test_set),
    )