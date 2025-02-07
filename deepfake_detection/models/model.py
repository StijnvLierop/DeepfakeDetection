from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def predict(self, instance) -> np.ndarray:
        raise NotImplementedError


    def get_features(self, instance) -> np.ndarray:
        raise NotImplementedError
