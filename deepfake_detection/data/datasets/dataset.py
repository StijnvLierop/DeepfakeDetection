from abc import ABC, abstractmethod
from typing import Iterable, Sized

from deepfake_detection.data.instance import Instance


class Dataset(ABC, Iterable[Instance], Sized):
    """
    An abstract dataset class that other datasets should inherit from.

    :param name: The name of the dataset.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.name)