from abc import ABC, abstractmethod
from typing import List, Union

from deepfake_detection.data import Instance, Dataset
from deepfake_detection.models.prediction import Prediction


class Model(ABC):

    def __init__(self, name):
        self.name = name

    def predict(self, instance: Instance) -> Prediction:
        return self.predict_batch([instance])[0]

    @abstractmethod
    def predict_batch(self, instances: Union[List[Instance], Dataset]) -> List[Prediction]:
        raise NotImplementedError