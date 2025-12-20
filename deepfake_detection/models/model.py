from abc import ABC, abstractmethod
from typing import List, Union, Any

from deepfake_detection.data import Instance, Dataset
from deepfake_detection.models.prediction import Prediction


class Model(ABC):
    """
    Abstract base class that every model should inherit from.
    """

    def __init__(self, name):
        self.name = name

    def predict(self, instance: Instance) -> Prediction:
        return self.predict_batch([instance])[0]

    @abstractmethod
    def predict_batch(self, instances: Union[List[Instance], Dataset]) -> List[Prediction]:
        raise NotImplementedError


class TrainableMixin(ABC):
    """
    Extension of the model class that defines methods to make a model trainable.
    """

    @abstractmethod
    def get_model_parameters(self) -> Any:
        """
        Returns the trainable parameters of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_pass(self, inputs: Any) -> Any:
        """
        Returns the output of a single forward pass through the model.
        """
        raise NotImplementedError

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """
        Saves the weights of the model to a given path.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        """
        Sets the model in train mode.
        """
        raise NotImplementedError
