from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict

import torch

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


class TrainableMixin(torch.nn.Module):
    """
    Extension of the model class that defines methods to make a model trainable.
    """

    @abstractmethod
    def forward(self, inputs: Any, labels: Any = None, **kwargs) -> Dict:
        """
        Returns the output of a single forward pass through the model in a dictionary.

        If labels are provided, this method should return the loss of the forward pass as well.
        """
        raise NotImplementedError
