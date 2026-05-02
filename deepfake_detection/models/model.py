from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict, Optional

import torch

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.data.instance import Instance
from deepfake_detection.models.prediction import Prediction


def load_model_wrapper(func):
    """
    Decorator that loads the model weights if they haven't been loaded yet.
    """
    def wrapper(self, *args, **kwargs):
        if not self.model_loaded:
            self.load_model()
        return func(self, *args, **kwargs)
    return wrapper


class Model(ABC):
    """
    Abstract base class that every model should inherit from.
    """

    def __init__(self, name: str, load_model: bool):
        """
        :param name: Name of the model.
        :param load_model: Whether to immediately load the model weights upon initialization.
        """
        self.name = name
        self.model_loaded = False

        # If load_model is True, immediately load the model
        if load_model:
            self.load_model()
            self.model_loaded = True

    @load_model_wrapper
    def predict(self, instance: Instance) -> Prediction:
        return self.predict_batch([instance])[0]

    def load_model(self):
        pass

    @abstractmethod
    @load_model_wrapper
    def predict_batch(
        self, instances: Union[List[Instance], Dataset]
    ) -> List[Prediction]:
        raise NotImplementedError


class TrainableMixin(torch.nn.Module, ABC):
    """
    Extension of the model class that defines methods to make a model trainable.
    """

    @property
    def data_collator(self):
        return None

    @abstractmethod
    def forward(self, inputs: Any, labels: Optional[Any] = None, **kwargs) -> Dict:
        """
        Returns the output of a single forward pass through the model in a dictionary.

        If labels are provided, this method should return the loss of the forward pass as well.
        """
        raise NotImplementedError

    @abstractmethod
    def transform_input(self, instance: Instance, **kwargs) -> torch.Tensor:
        """
        Transforms an input instance into a tensor that can be used as input to the model.
        """
        raise NotImplementedError