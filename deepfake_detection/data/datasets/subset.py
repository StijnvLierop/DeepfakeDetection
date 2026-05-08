import numpy as np

from deepfake_detection.data.dataset import MapStyleDatasetMixin
from deepfake_detection.data.dataset import Dataset


class SubsetDataset(MapStyleDatasetMixin, Dataset):
    """
    Returns a subset of a dataset using a given list of indices.

    :param dataset: The dataset to take the subset of.
    :param indices: The indices of the original dataset to include in the subset.
    :return: A subset dataset.
    """

    def __init__(
        self, dataset: MapStyleDatasetMixin, indices: np.ndarray, dataset_name: str
    ):
        super().__init__(dataset_name=dataset_name)
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        # Maps the requested index (0 to len(subset)) to the original dataset index
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
