from typing import Sequence, Optional

from deepfake_detection.data.dataset import Dataset, MapStyleDatasetMixin
from deepfake_detection.data.instance import Instance


class ListDataset(MapStyleDatasetMixin, Dataset):
    """
    A dataset class for datasets that can be initialized using a sequence of instances.
    """

    def __init__(
        self, instances: Sequence[Instance], dataset_name: Optional[str] = None
    ):
        super().__init__(dataset_name=dataset_name)
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        return self.instances[idx]
