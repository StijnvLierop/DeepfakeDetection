from abc import ABC, abstractmethod
from typing import List, Union

from data import Instance, Dataset
from deepfake_detection.models.prediction import Prediction


class Model(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def predict(self, instance: Instance) -> Prediction:
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, instances: Union[List[Instance], Dataset]) -> List[Prediction]:
        raise NotImplementedError