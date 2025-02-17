import os
from typing import Iterable

from deepfake_detection.data.datasets.dataset import Dataset
from deepfake_detection.data.datasets.instance import ImageInstance


class FileImageDataset(Dataset):
    """
    This dataset loads a dataset of images from a filesystem.
    The dataset should be stored on the filesystem as follows:

    <root dataset dir>
        fake
            <label 1>
            - image 1
            - image 2
        camera1
            <label 2>
            - image 1
            - image 2
        ...

    Non-image files are ignored.

    :param path: The path to the root folder of the dataset.
    :param name: The name of the dataset.
    """

    def __init__(self, path : str, name : str):
        super(FileImageDataset, self).__init__(name=name)
        self.path = path

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        n = 0
        # Loop over folders (authenticity class) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                for subfolder in os.listdir(os.path.join(self.path, folder)):
                    # If directory
                    if os.path.isdir(os.path.join(self.path, folder, subfolder)):
                        # Loop over images
                        for img in os.listdir(os.path.join(self.path, folder, subfolder)):
                            if img.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
                                n += 1
        return n

    def __iter__(self) -> Iterable[ImageInstance]:
        # Loop over folders (labels) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                for subfolder in os.listdir(os.path.join(self.path, folder)):
                    # If directory
                    if os.path.isdir(os.path.join(self.path, folder, subfolder)):
                        # Loop over images
                        for img in os.listdir(os.path.join(self.path, folder, subfolder)):
                            if img.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
                                yield ImageInstance(os.path.join(self.path, folder, subfolder, img),
                                                    subfolder,
                                                    folder)
                            else:
                                print("Found file that is not a jpg, jpeg or png file: {}".format(img))