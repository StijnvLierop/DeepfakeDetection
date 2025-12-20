from typing import Tuple

import numpy as np
import torch

from deepfake_detection.data import Instance, ImageInstance, FileImageInstance
from deepfake_detection.data.dataset import Dataset, MapStyleDatasetMixin


def instance_to_torch_dict(instance: Instance) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function transforms an Instance to a dict that can be loaded by a Pytorch DataLoader.
    """
    # Transform instance data to tensor
    if isinstance(instance, ImageInstance) or isinstance(instance, FileImageInstance):
        input_tensor = torch.tensor(np.array(instance.data), dtype=torch.uint8).permute(2, 0, 1)
    # TODO: add audio and video instance
    else:
        raise ValueError(f"Instance type {type(instance)} cannot be converted to torch tensor.")

    return input_tensor, torch.tensor(instance.annotation.binary_label, dtype=torch.int8)


class TorchDataset(torch.utils.data.Dataset):
    """
    A wrapper to load a dataset as a Pytorch Map-style dataset.
    """

    def __init__(self, dataset: MapStyleDatasetMixin):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, idx: int):
        return instance_to_torch_dict(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


class TorchIterableDataset(torch.utils.data.IterableDataset):
    """
    A wrapper to load a dataset as a Pytorch iterable dataset.
    """

    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for instance in self.dataset:
            yield instance_to_torch_dict(instance)
