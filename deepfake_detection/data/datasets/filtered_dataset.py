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
        for idx, instance in enumerate(self.dataset):
            if idx in self.indices:
                yield instance

    def __len__(self):
        return len(list(self.indices))

    def __getitem__(self, idx: int):
        return self.dataset[list(self.indices)[idx]]