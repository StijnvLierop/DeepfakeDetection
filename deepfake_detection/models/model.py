from abc import ABC, abstractmethod

import numpy as np

from deepfake_detection.data.datasets.dataset import Dataset


class Model(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def predict(self, instance) -> np.ndarray:
        raise NotImplementedError

    def train(self, train_dataset: Dataset):
        return "This method was not implemented for this model."