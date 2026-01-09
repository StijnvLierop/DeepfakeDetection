import itertools
from typing import Iterable, Sized

from deepfake_detection.data.dataset import Dataset


class CombinedDataset(Dataset):
    """
    Helper class that can be used to combine multiple datasets into a single dataset.
    """

    def __init__(self, datasets: Iterable[Dataset]):
        """
        :param datasets: The datasets to combine.
        """
        combined_name = 'combined_' + '_'.join(dataset.name if dataset.name is not None else '' for dataset in datasets)
        super().__init__(name=combined_name)
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