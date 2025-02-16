import os
from abc import ABC
from functools import cached_property
from typing import List

from PIL import Image
import cv2


class Instance(ABC):

    def __init__(self, path: str, class_label: str, authenticity_label: str):
        self.path = path
        self.class_label = class_label
        self.authenticity_label = authenticity_label

class ImageInstance(Instance):

    @cached_property
    def data(self):
        return Image.open(self.path).convert('RGB')

class ImageSequenceInstance(Instance):

    @cached_property
    def data(self) -> List[ImageInstance]:
        return [ImageInstance(os.path.join(self.path, img), self.class_label, self.authenticity_label)
                for img in os.listdir(self.path)]

    def __len__(self):
        return len(self.data)

class VideoInstance(Instance):

    @cached_property
    def data(self):
        return cv2.VideoCapture(self.path)