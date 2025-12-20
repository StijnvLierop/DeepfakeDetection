from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from deepfake_detection.data import Dataset
from deepfake_detection.models.model import TrainableMixin


@dataclass
class TrainParams:
    """
    Class that acts as a container for any training parameters.
    """
    # Essential parameters
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 8
    optimizer: str = "adam"
    criterion: str = "cross_entropy"

    # Dataset/Dataloader settings
    shuffle: bool = True
    num_workers: int = 1
    pin_memory: bool = True

    # Hardware & Precision
    device: str = 'cuda'

    # Logging & Storage
    output_path: str = "checkpoints/"
    save_every_epoch: bool = False


class Trainer(ABC):
    """
    Abstract class that represents a trainer that can train a model on a dataset.
    """

    def __init__(self, model: TrainableMixin, train_params: TrainParams):
        """
        :param model: The model to train.
        :param train_params: The training parameters.
        """
        self.model = model
        self.train_params = train_params

    @abstractmethod
    def train(self, dataset: Dataset):
        """
        Trains a model on a dataset.

        :param dataset: The dataset to train the model on.
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str):
        """
        Saves the current model checkpoint to the given path.
        """
        raise NotImplementedError
