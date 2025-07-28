from typing import Iterable

from deepfake_detection.data import Dataset


class FilteredDataset(Dataset):
    """
    Filters a given dataset using a list of indices.

    :param dataset: The dataset to filter.
    :param indices: The indices to filter by.
    :return: A filtered dataset.
    """

    def __init__(self, dataset: Dataset, indices: Iterable[int]):
        super().__init__(dataset.name)
        self.dataset = dataset
        self.indices = indices

    def __iter__(self):
        for i in self.dataset:
            if i in self.indices:
                yield i