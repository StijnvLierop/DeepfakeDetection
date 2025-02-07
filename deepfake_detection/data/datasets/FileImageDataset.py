import os
from typing import Iterable

from deepfake_detection.data.datasets.dataset import Dataset
from deepfake_detection.data.datasets.instance import ImageInstance


class FileImageDataset(Dataset):
    """
    This dataset loads a dataset of images from a filesystem.
    The dataset should be stored on the filesystem as follows:

    <root dataset dir>
        <label 1>
            - image 1
            - image 2
        <label 2>
            - image 1
            - image 2
        ...

    Non-image files are ignored.

    :param path: The path to the root folder of the dataset.
    :param name: The name of the dataset.
    """

    def __init__(self, path, name):
        super(FileImageDataset, self).__init__(name=name)
        self.path = path

    def __len__(self):
        """
        Returns a the length of the dataset.
        """
        n = 0
        # Loop over folders (labels) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                # Loop over images
                for img in os.listdir(os.path.join(self.path, folder)):
                    if img.split('.')[-1] in ['jpg', 'jpeg', 'png']:
                        n += 1
        return n

    def __iter__(self) -> Iterable[ImageInstance]:
        # Loop over folders (labels) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                # Loop over images
                for img in os.listdir(os.path.join(self.path, folder)):
                    # if image
                    if img.split('.')[-1] in ['jpg', 'jpeg', 'png']:
                        yield ImageInstance(os.path.join(self.path, folder, img), folder)