import os
from typing import Iterable

from deepfake_detection.data.datasets.dataset import Dataset
from deepfake_detection.data.datasets.instance import ImageSequenceInstance


class FileImageSequenceDataset(Dataset):
    """
    This dataset loads a dataset of images from a filesystem.
    The dataset should be stored on the filesystem as follows:

    <root dataset dir>
        <label 1>
            - <image sequence 1>
                - frame 1
                - frame 2
            - <image sequence 2>
                - frame 1
                - frame 2
        <label 2>
            - <image sequence 1>
                - frame 1
                - frame 2
            - <image sequence 2>
                - frame 1
                - frame 2
        ...

    Non-image files are ignored.

    :param path: The path to the root folder of the dataset.
    :param name: The name of the dataset.
    """

    def __init__(self, name, path):
        super().__init__(name)
        self.path = path

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        n = 0
        # Loop over folders (models) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                # Loop over subdirs with image sequences inside them
                for img_folder in os.listdir(os.path.join(self.path, folder)):
                    # if folder
                    if os.path.isdir(os.path.join(self.path, folder, img_folder)):
                        n += 1
        return n

    def __iter__(self) -> Iterable[ImageSequenceInstance]:
        # Loop over folders (models) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                # Loop over image sequences
                for img_folder in os.listdir(os.path.join(self.path, folder)):
                    # if folder
                    if os.path.isdir(os.path.join(self.path, folder, img_folder)):
                        yield ImageSequenceInstance(os.path.join(self.path, folder, img_folder), folder)