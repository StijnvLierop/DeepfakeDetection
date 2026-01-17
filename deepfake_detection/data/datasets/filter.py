from typing import Callable

from deepfake_detection.data.instance import Instance
from deepfake_detection.data.dataset import Dataset


class FilteredDataset(Dataset):
    """
    Filters a dataset using a given filter function.

    :param dataset: The dataset to filter.
    :param filter_func: The function to use for filtering. Should take an instance as input and return True or False.
    :return: A filtered dataset.
    """

    def __init__(self, dataset: Dataset, filter_func: Callable[[Instance], bool]):
        super().__init__(name=None if dataset.name is None else f"filtered_{dataset.name}")
        self.dataset = dataset
        self.filter_func = filter_func

    def __iter__(self):
        for idx, instance in enumerate(self.dataset):
            if self.filter_func(instance):
                yield instance

    def __len__(self):
        return len(list(self.__iter__()))
