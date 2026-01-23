from abc import ABC, abstractmethod
from itertools import islice
from typing import Iterable, Optional, Callable
from collections import Counter

from deepfake_detection.data.instance import Instance


class Dataset(ABC, Iterable[Instance]):
    """
    An abstract dataset class that other datasets should inherit from.
    """

    def __init__(self, name: Optional[str] = 'unspecified_dataset'):
        """
        :param name: The name of the dataset (optional).
        """
        self.name = name

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

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

        :param attribute: The attribute to count. Must be 'authenticity_label', 'source_label' or a key in 'meta'.
        """
        # Count occurrences of each authenticity label
        if attribute == 'authenticity_label' or attribute == 'source_label':
            counts = Counter(instance.annotation.get_label(attribute) for instance in self)
        else:
            counts = Counter(instance.meta[attribute] for instance in self)
        return counts
            
    def info(self, attribute: str = 'authenticity_label') -> str:
        """
        Returns a string representing the number of samples per label in the dataset.

        :param attribute: The attribute to count. Must be 'authenticity_label', 'source_label' or a key in 'meta'.
        """
        # Calculate counts
        counts = self.count_labels(attribute)

        # Format the header
        header = f"Dataset: {self.name}\n"
        separator = "-" * (len(header) - 1) + "\n"

        # Create lines for each label, sorted by count (descending)
        stats = "\n".join([f"{label}: {count}" for label, count in counts.most_common()])

        return f"{header}{separator}{stats}"

    def filter(self, func: Callable[[Instance], bool]) -> 'Dataset':
        """
        Returns a new dataset containing only the instances for which the given function returns True.

        :param func: A function that takes an instance and returns a boolean.
        :return: A new dataset containing only the instances for which the given function returns True.
        """
        from deepfake_detection.data.datasets.filter import FilteredDataset
        return FilteredDataset(self, func)


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
