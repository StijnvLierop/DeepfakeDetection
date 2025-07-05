import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sized

from deepfake_detection.data.instance import Instance
from deepfake_detection.utils.serialization import InstanceEncoder


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

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def __eq__(self, other):
        return set(self.__iter__()) == set(other.__iter__())

    def save(self, path: Path) -> None:
        """
        Serialize this dataset to JSON and save it to the specified path.

        :param path: The path to the JSON file where the data should be stored.
        """
        # Open file write path
        with open(path, "w") as outfile:

            # Get all instances
            instances = list(self.__iter__())

            # Encode to JSON
            json.dump(instances, outfile, cls=InstanceEncoder)