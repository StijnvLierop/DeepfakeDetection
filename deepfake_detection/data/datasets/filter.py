from typing import Callable

from deepfake_detection.data.dataset import Dataset


class FilteredDataset(Dataset):
    """
    Filters a dataset using a given filter function.

    :param dataset: The dataset to filter.
    :param filter_func: The function to use for filtering. Should take a dataset as input and return a list of indices.
    :return: A filtered dataset.
    """

    def __init__(self, dataset: Dataset, filter_func: Callable[[Dataset], list[int]]):
        super().__init__(name=None if dataset.name is None else f"{dataset.name}")
        self.dataset = dataset
        self.indices = filter_func(dataset)

    def __iter__(self):
        for idx, instance in enumerate(self.dataset):
            if idx in self.indices:
                yield instance

    def __len__(self):
        return len(self.indices)
