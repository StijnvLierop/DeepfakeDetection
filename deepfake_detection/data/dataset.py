from abc import ABC, abstractmethod
from itertools import islice
from typing import Iterable, Optional, Callable, Any
from collections import Counter

import numpy as np

from deepfake_detection.data.instance import Instance


class Dataset(ABC, Iterable[Instance]):
    """
    An abstract dataset class that other datasets should inherit from.
    """

    def __init__(self, dataset_name: Optional[str] = 'unspecified_dataset'):
        """
        :param dataset_name: The name of the dataset (optional).
        """
        self.dataset_name = dataset_name

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other):
        return set(self.__iter__()) == set(other.__iter__())

    def iter(self, batch_size: int):
        iterator = iter(self)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            yield batch

    def count_labels(self, attribute: str) -> Counter:
        """
        Returns a Counter containing counts for every label in a given attribute.

        :param attribute: The attribute to count. Must be a key in 'Annotation'.
        """
        counts = Counter(instance.annotation.get_label(attribute)
                         if instance.annotation else '<no annotation>' for instance in self)
        return counts
            
    def info(self, attribute: str = 'authenticity') -> str:
        """
        Returns a string representing the number of samples per label in the dataset.

        :param attribute: The attribute to count. Must be a key in 'Annotation'.
        """
        # Calculate counts
        counts = self.count_labels(attribute)

        # Format the header
        header = f"Dataset: {self.dataset_name}\n"
        separator = "-" * (len(header) - 1) + "\n"

        # Create lines for each label, sorted by count (descending)
        stats = "\n".join([f"{label}: {count}" for label, count in counts.most_common()]) + "\n"

        return f"{header}{separator}{stats}{separator}"

    def filter(self, func: Callable[[Instance], bool]) -> 'Dataset':
        """
        Returns a new dataset containing only the instances for which the given function returns True.

        :param func: A function that takes an instance and returns a boolean.
        :return: A new dataset containing only the instances for which the given function returns True.
        """
        from deepfake_detection.data.datasets import FilteredDataset
        return FilteredDataset(self, func)

    def map(self, func: Callable[[Instance], Instance]) -> 'Dataset':
        """
        Returns a new dataset where the mapping function has been applied to each instance.

        :param func: A function that takes an instance and returns a new instance.
        :return: A new dataset with the function applied to each instance.
        """
        from deepfake_detection.data.datasets import MappedDataset
        return MappedDataset(self, func)


class MapStyleDatasetMixin(ABC):
    """
    Extension of the regular dataset class that adds the __getitem__ and __len__ methods.
    In addition, it automatically provides an implementation for the __iter__ method.
    """

    @abstractmethod
    def __getitem__(self, idx: int) -> Instance:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]