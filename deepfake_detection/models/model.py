from abc import ABC, abstractmethod

from deepfake_detection.models.prediction import Prediction


class Model(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def predict(self, instance) -> Prediction:
        raise NotImplementedError


class TrainableModel(Model, ABC):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def trainable_model(self):
        raise NotImplementedError

    @abstractmethod
    def prepare_for_training(self) -> None:
        raise NotImplementedError