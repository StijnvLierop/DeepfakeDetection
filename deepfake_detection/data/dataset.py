from abc import ABC, abstractmethod
from itertools import islice
from typing import Iterable, Optional

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
