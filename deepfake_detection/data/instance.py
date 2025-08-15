import os
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Optional

from PIL import Image
import cv2

from deepfake_detection.data.annotation import Annotation
from deepfake_detection.utils.hashing import hash_image_to_int


class Instance(ABC):
    """
    A single dataset instance. Each instance can have one or more optional labels.

    :param annotation: An annotation for the instance.
    """
    def __init__(self, annotation: Optional[Annotation]=None):
        self.annotation = annotation

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
    :param annotation: An annotation for the instance.
    """

    def __init__(self, data: Image, annotation: Optional[Annotation]=None):
        super().__init__(annotation)
        self.data = data

    def __hash__(self):
        return hash_image_to_int(self.data)

class FileImageInstance(Instance):
    """
    An image instance that is created from an image file stored on disk.

    :param path: The image path.
    :param annotation: An annotation for the instance.
    """

    def __init__(self, path: str, annotation: Optional[Annotation]=None):
        super().__init__(annotation)
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
    :param annotation: An annotation for the instance.
    """

    def __init__(self, path: str, annotation: Optional[Annotation]=None):
        super().__init__(annotation)
        self.path = Path(path)

    @cached_property
    def data(self) -> list[FileImageInstance]:
        return [FileImageInstance(os.path.join(self.path, img), self.annotation)
                for img in os.listdir(self.path)]

    def __len__(self):
        return len(self.data)

    def __hash__(self):
        return hash(self.path)

class FileVideoInstance(Instance):
    """
    An instance that is created from a video stored on disk.

    :param path: The path to the video file.
    :param annotation: An annotation for the instance.
    """

    def __init__(self, path: str, annotation: Optional[Annotation]=None):
        super().__init__(annotation)
        self.path = Path(path)

    @cached_property
    def data(self):
        return cv2.VideoCapture(str(self.path))

    def __hash__(self):
        return hash(self.path)