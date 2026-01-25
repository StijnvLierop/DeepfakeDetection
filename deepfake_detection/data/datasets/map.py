from typing import Callable, Sized

from deepfake_detection.data.instance import Instance
from deepfake_detection.data.dataset import Dataset


class MappedDataset(Dataset):
    """
    Applies a given function to each instance in the dataset.

    :param dataset: The dataset to apply the function to.
    :param mapping_func: The function to apply to each instance.
    :return: A mapped dataset.
    """

    def __init__(self, dataset: Dataset, mapping_func: Callable[[Instance], Instance]):
        super().__init__(dataset_name=dataset.dataset_name)
        self.dataset = dataset
        self.mapping_func = mapping_func

    def __getitem__(self, idx):
        instance = self.dataset[idx]
        return self.mapping_func(instance.deepcopy())

    def __len__(self):
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        else:
            raise ValueError("Cannot determine length of unsized dataset.")