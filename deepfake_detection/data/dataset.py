from abc import ABC, abstractmethod
from itertools import islice
from typing import Iterable, Sized, List, Generator

from deepfake_detection.data.instance import Instance


class Dataset(ABC, Iterable[Instance], Sized):
    """
    An abstract dataset class that other datasets should inherit from.

    :param name: The name of the dataset (optional).
    """

    def __init__(self, name: str = None):
        self.name = name

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return len(list(self.__iter__()))

    def __eq__(self, other):
        return set(self.__iter__()) == set(other.__iter__())

    def iter(self, batch_size: int):
        iterator = iter(self)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            yield batch