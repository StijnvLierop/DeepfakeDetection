import os
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path

from PIL import Image
import cv2
import hashlib

from deepfake_detection.utils.hashing import hash_image_to_int


class Instance(ABC):
    """
    A single dataset instance. Each instance has an optional label.

    :param label: An optional instance label.
    """
    def __init__(self, label: str = None):
        self.label = label

    def __eq__(self, other):
        if not isinstance(other, Instance):
            return ValueError("Other object is not of type Instance.")
        return self.__hash__() == other.__hash__()

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError

class ImageInstance(Instance):
    """
    A default image instance is initialized with only image data and an optional label.

    :param data: The image data in the form of a PIL Image.
    :param label: An optional instance label.
    """

    def __init__(self, data: Image, label: str = None):
        super().__init__(label)
        self.data = data

    def __hash__(self):
        return hash_image_to_int(self.data)

class FileImageInstance(Instance):
    """
    An image instance that is created from an image file stored on disk.
    """

    def __init__(self, path: str, label: str = None):
        super().__init__(label)
        self.path = Path(path)

    @cached_property
    def data(self):
        return Image.open(self.path).convert('RGB')

    def __hash__(self):
        return hash(self.path)

class FileImageSequenceInstance(Instance):
    """
    An instance that is created from a sequence of images stored on disk.

    :param path: The path to the image sequence directory.
    :param label: An optional instance label.
    """

    def __init__(self, path: str, label: str = None):
        super().__init__(label)
        self.path = Path(path)

    @cached_property
    def data(self) -> list[FileImageInstance]:
        return [FileImageInstance(os.path.join(self.path, img), self.label)
                for img in os.listdir(self.path)]

    def __len__(self):
        return len(self.data)

    def __hash__(self):
        return hash(self.path)

class FileVideoInstance(Instance):
    """
    An instance that is created from a video stored on disk.

    :param path: The path to the video file.
    :param label: An optional instance label.
    """

    def __init__(self, path: str, label: str = None):
        super().__init__(label)
        self.path = Path(path)

    @cached_property
    def data(self):
        return cv2.VideoCapture(str(self.path))

    def __hash__(self):
        return hash(self.path)