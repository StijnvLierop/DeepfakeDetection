from abc import ABC, abstractmethod

from deepfake_detection.models.prediction import Prediction


class Model(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def predict(self, instance) -> Prediction:
        raise NotImplementedError