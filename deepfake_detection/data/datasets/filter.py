from typing import Callable

from deepfake_detection.data.dataset import MapStyleDatasetMixin
from deepfake_detection.data.instance import Instance
from deepfake_detection.data.dataset import Dataset


class FilteredDataset(MapStyleDatasetMixin, Dataset):
    """
    Filters a dataset using a given filter function.

    :param dataset: The dataset to filter.
    :param filter_func: The function to use for filtering. Should take an instance as input and return True or False.
    :return: A filtered dataset.
    """

    def __init__(self, dataset: Dataset, filter_func: Callable[[Instance], bool]):
        super().__init__(dataset_name=None if dataset.dataset_name is None else f"filtered_{dataset.dataset_name}")
        self.dataset = dataset
        self.filter_func = filter_func

        # Pre-calculate the valid indices once
        self.valid_indices = [
            idx for idx, instance in enumerate(dataset)
            if filter_func(instance)
        ]

    def __getitem__(self, idx):
        # Map the requested index to the original dataset's index
        original_idx = self.valid_indices[idx]
        return self.dataset[original_idx]

    def __len__(self):
        return len(self.valid_indices)
