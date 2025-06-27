import torch
from PIL import Image

from deepfake_detection.data.datasets.dataset import Dataset
from deepfake_detection.utils.labels import encode_label


class TrainDataset(torch.utils.data.Dataset):
    """
    A wrapper dataset class used to make datasets suitable for training.
    Currently, this dataset only supports training with binary labels.

    :param base_dataset: Base dataset to use.
    :param transform: Transform to apply on all samples.
    """

    def __init__(self, base_dataset: Dataset, transform=None):
        super(TrainDataset, self).__init__()

        # Set base variables
        self.base_dataset = base_dataset
        self.transform = transform
        self.labels = None

        # Index all samples in dataset
        self.samples = [(x.path, x.label) for x in base_dataset]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index):
        # Get path and label
        path, label = self.samples[index]

        # Transform label to binary
        label = encode_label(label)

        # Open data
        data = Image.open(path).convert('RGB')

        # Transform if needed
        if self.transform:
            data = self.transform(data)

        return data, label