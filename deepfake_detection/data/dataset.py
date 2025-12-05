import json
from abc import ABC, abstractmethod
from json import JSONEncoder
from pathlib import Path
from typing import Iterable, Sized

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