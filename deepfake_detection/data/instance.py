import os
from abc import ABC
from functools import cached_property
from pathlib import Path
from typing import List

from PIL import Image
import cv2


class Instance(ABC):
    """
    A single dataset instance. Instances are identified by their path. Additionally, each instance has a label.

    :param path: The path to the instance.
    :param label: The label of the instance.
    """
    def __init__(self, path: str, label: str):
        self.path = Path(path)
        self.label = label

    def __eq__(self, other):
        if not isinstance(other, Instance):
            return ValueError("Other object is not of type Instance.")
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        """
        Because data are retrieved dynamically, dataset instances are identified only by path.
        """
        return hash(self.path)

class ImageInstance(Instance):

    @cached_property
    def data(self):
        return Image.open(self.path).convert('RGB')

class ImageSequenceInstance(Instance):

    @cached_property
    def data(self) -> List[ImageInstance]:
        return [ImageInstance(os.path.join(self.path, img), self.label)
                for img in os.listdir(self.path)]

    def __len__(self):
        return len(self.data)

class VideoInstance(Instance):

    @cached_property
    def data(self):
        return cv2.VideoCapture(self.path)