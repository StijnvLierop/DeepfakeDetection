import os
from typing import Iterable

from deepfake_detection.data.datasets.dataset import Dataset
from deepfake_detection.data.datasets.instance import ImageInstance


class FileImageDataset(Dataset):

    def __init__(self, path, name):
        super(FileImageDataset, self).__init__(name=name)
        self.path = path

    def __len__(self):
        n = 0
        # Loop over folders (models) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                # Loop over images
                for img in os.listdir(os.path.join(self.path, folder)):
                    # if image
                    if img.split('.')[-1].lower() in ['jpg', 'png', 'bmp', 'jpeg', 'gif']:
                        n += 1
        return n

    def __iter__(self) -> Iterable[ImageInstance]:
        # Loop over folders (models) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                # Loop over images
                for img in os.listdir(os.path.join(self.path, folder)):
                    # if image
                    if img.split('.')[-1].lower() in ['jpg', 'png', 'bmp', 'jpeg', 'gif']:
                        yield ImageInstance(os.path.join(self.path, folder, img), folder)