from typing import Union, Callable, Any, Optional

import torch
from torchvision.transforms import v2

from deepfake_detection.data.dataset import Dataset


def _build_item(
    instance, inputs: torch.Tensor, label: Optional[str], pos_label: Optional[str]
) -> dict:
    item = {"inputs": inputs}
    if label is not None:
        item["labels"] = torch.tensor(
            instance.annotation.get_label(label) == pos_label,
            dtype=torch.long,
        )
    if instance.annotation is not None and instance.annotation.mask is not None:
        h, w = inputs.shape[-2], inputs.shape[-1]
        mask_tensor = v2.functional.resize(
            v2.functional.pil_to_tensor(instance.annotation.mask),
            [h, w],
            interpolation=v2.InterpolationMode.NEAREST,
        )
        item["masks"] = (mask_tensor.squeeze(0) > 128).long()
    return item


class TorchDataset(torch.utils.data.Dataset):
    """
    A wrapper to load a dataset as a Pytorch Map-style dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Union[v2.Compose, Callable[[Any], torch.Tensor]],
        label: Optional[str],
        pos_label: Optional[str],
    ):
        super().__init__()
        self.dataset = dataset
        self.transforms = transform
        self.label = label
        self.pos_label = pos_label

    def __getitem__(self, idx: int):
        instance = self.dataset[idx]
        return _build_item(
            instance, self.transforms(instance), self.label, self.pos_label
        )

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
        label: Optional[str],
        pos_label: Optional[str],
    ):
        super().__init__()
        self.dataset = dataset
        self.transforms = transform
        self.label = label
        self.pos_label = pos_label

    def __iter__(self):
        for instance in self.dataset:
            yield _build_item(
                instance, self.transforms(instance), self.label, self.pos_label
            )
