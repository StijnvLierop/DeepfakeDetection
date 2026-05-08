from typing import Union, Callable, Any

import torch
from torchvision.transforms import v2

from deepfake_detection.data.dataset import Dataset, MapStyleDatasetMixin


class TorchDataset(torch.utils.data.Dataset):
    """
    A wrapper to load a dataset as a Pytorch Map-style dataset.
    """

    def __init__(
        self,
        dataset: MapStyleDatasetMixin,
        transform: Union[v2.Compose, Callable[[Any], torch.Tensor]],
        label: str,
        pos_label: str,
    ):
        super().__init__()
        self.dataset = dataset
        self.transforms = transform
        self.label = label
        self.pos_label = pos_label

    def __getitem__(self, idx: int):
        instance = self.dataset[idx]
        return {
            "inputs": self.transforms(instance),
            "labels": torch.tensor(
                instance.annotation.get_label(self.label) == self.pos_label,
                dtype=torch.long,
            ),
        }

    def __len__(self):
        return len(self.dataset)


class TorchIterableDataset(torch.utils.data.IterableDataset):
    """
    A wrapper to load a dataset as a Pytorch iterable dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Union[v2.Compose, Callable[[Any], torch.Tensor]],
        label: str,
        pos_label: str,
    ):
        super().__init__()
        self.dataset = dataset
        self.transforms = transform
        self.label = label
        self.pos_label = pos_label

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for instance in self.dataset:
            yield {
                "inputs": self.transforms(instance),
                "labels": torch.tensor(
                    instance.annotation.get_label(self.label) == self.pos_label,
                    dtype=torch.long,
                ),
            }
