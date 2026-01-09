import itertools
from typing import Iterable, Sized, Optional

from deepfake_detection.data.dataset import Dataset


class CombinedDataset(Dataset):
    """
    Helper class that can be used to combine multiple datasets into a single dataset.
    """

    def __init__(self, datasets: Iterable[Dataset], dataset_name: Optional[str] = None):
        """
        :param datasets: The datasets to combine.
        :param dataset_name: The name of the combined dataset.
        """
        super().__init__(name=dataset_name)
        self.datasets = datasets

    def __iter__(self):
        return itertools.chain(*self.datasets)

    def __len__(self):
        n = 0
        for dataset in self.datasets:
            if isinstance(dataset, Sized):
                n += len(dataset)
            else:
                raise ValueError("Dataset does not support __len__")
        return n