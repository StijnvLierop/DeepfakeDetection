import os
from typing import Iterable

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.data.instance import ImageSequenceInstance


class FileImageSequenceDataset(Dataset):
    """
    This dataset loads a dataset of images from a filesystem.
    The dataset should be stored on the filesystem as follows:

    <root dataset dir>
        real
            <label 1>
                - <image sequence 1>
                    - frame 1
                    - frame 2
                - <image sequence 2>
                    - frame 1
                    - frame 2
        fake
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
    :param split_file: The path to a file containing the directory names of the image sequences that should be returned.
    """

    def __init__(self, path: str, name: str = None, split_file: str = None):
        super().__init__(name)
        self.path = path

        # If split file provided store the filenames of included instances in a list
        if split_file:
            with open(split_file, 'r') as f:
                self.included_instances = f.read().splitlines()
        else:
            self.included_instances = None

    def __iter__(self) -> Iterable[ImageSequenceInstance]:
        # Loop over folders (models) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                # Loop over folders (models) in dataset
                for subfolder in os.listdir(os.path.join(self.path, folder)):
                    # If directory
                    if os.path.isdir(os.path.join(self.path, folder, subfolder)):
                        # Loop over subdirs with image sequences inside them
                        for img_folder in os.listdir(os.path.join(self.path, folder, subfolder)):
                            # if folder
                            if os.path.isdir(os.path.join(self.path, folder, subfolder, img_folder)):
                                if self.included_instances is None or img_folder in self.included_instances:
                                    yield ImageSequenceInstance(os.path.join(self.path, folder, subfolder, img_folder),
                                                                subfolder)