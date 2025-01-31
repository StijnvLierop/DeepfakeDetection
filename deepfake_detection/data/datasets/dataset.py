from abc import ABC, abstractmethod
from typing import Iterable

from deepfake_detection.data.datasets.instance import Instance


class Dataset(ABC, Iterable[Instance]):
    """
    An abstract dataset class that other datasets should inherit from.

    :param name: The name of the dataset.
    """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError