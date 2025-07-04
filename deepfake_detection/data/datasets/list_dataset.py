from typing import Sequence

from deepfake_detection.data.datasets.dataset import Dataset
from deepfake_detection.data.instance import Instance


class ListDataset(Dataset):
    """
    A dataset class for datasets that can be initialized using a sequence of instances.
    """

    def __init__(self, name: str, instances: Sequence[Instance]):
        super().__init__(name)
        self.instances = instances

    def __iter__(self):
        return iter(self.instances)

    def __len__(self):
        return len(self.instances)
