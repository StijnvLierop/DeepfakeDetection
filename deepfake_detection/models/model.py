from abc import ABC, abstractmethod
from typing import List, Union

from deepfake_detection.data import Instance, Dataset
from deepfake_detection.models.prediction import Prediction


class Model(ABC):

    def __init__(self, name):
        self.name = name

    def load_model(self):
        pass

    @abstractmethod
    def predict(self, instance: Instance) -> Prediction:
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, instances: Union[List[Instance], Dataset]) -> List[Prediction]:
        raise NotImplementedError


def lazy_loader(func):
    """
    Decorator function that can be added to a 'predict' or 'predict_batch' method of a Model class to load
    the model when this method is called.
    """
    def wrapper(self, *args, **kwargs):
        if self.model is None:
            self.load_model()
        return func(self, *args, **kwargs)
    return wrapper