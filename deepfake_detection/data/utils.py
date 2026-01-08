from typing import Union, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.data.datasets import FilteredDataset, ListDataset


def split_dataset(
    dataset: Dataset,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    shuffle: Optional[bool] = True,
    stratify: Optional[bool] = True,
    label_type: Optional[str] = "source_label",
) -> Tuple[ListDataset, ListDataset]:
    """
    Splits a dataset into training and test sets.

    :param dataset: Dataset to split.
    :param test_size: Float specifying the fraction of instances in the test set.
                      The remainder will be in the train set.
    :param random_state: Integer specifying the random state for splitting the dataset.
    :param shuffle: Boolean specifying whether to shuffle the dataset before splitting.
                    If shuffle=False then stratify must be None.
    :param stratify: If set to True, data is split in a stratified fashion using the class labels type in 'label_type'.
    :param label_type: If stratify=True, this specifies the type of labels to use for stratification.
                       Can be one of: 'authenticity_label', 'binary_label' and 'source_label'.
    """
    # Create index array
    index_array = np.arange(0, len(dataset))

    # Get labels
    labels = [i.annotation.get_label(label_type) for i in dataset] if stratify else None

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
    dataset1_name = dataset.name + "_split1" if dataset.name else "split1"
    dataset2_name = dataset.name + "_split2" if dataset.name else "split2"

    return (
        ListDataset(name=dataset1_name, instances=train_instances),
        ListDataset(name=dataset2_name, instances=test_instances),
    )


def sample_n_per_class(
    dataset: Dataset,
    n: int,
    label_type: Optional[str] = "source_label",
    random_seed: Optional[Union[int, None]] = None,
) -> FilteredDataset:
    """
    Samples n instances from each source class in the given dataset.
    If n is larger than the number of instances in a class, the maximum number of instances will be returned.

    :param dataset: The dataset to sample from.
    :param n: The number of instances to sample from each class.
    :param label_type: The label type in 'Annotation' to use for sampling.
                       Can be 'source_label' or 'authenticity_label'. Defaults to 'source_label'.
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

    # Return an iterable that contains n indices
    return FilteredDataset(dataset, indices)
