import os
from abc import ABC
from functools import cached_property
from typing import List

from PIL import Image


class Instance(ABC):

    def __init__(self, path, label):
        self.path = path
        self.label = label

class ImageInstance(Instance):

    @cached_property
    def data(self):
        return Image.open(self.path).convert('RGB')

class VideoInstance(Instance):

    @cached_property
    def data(self):
        raise NotImplementedError

class AudioInstance(Instance):

    @cached_property
    def data(self):
        raise NotImplementedError

class ImageSequenceInstance(Instance):

    @cached_property
    def data(self) -> List[ImageInstance]:
        return [ImageInstance(os.path.join(self.path, img), self.label) for img in os.listdir(self.path)]