import os
from functools import cached_property
from typing import Iterable

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.data.instance import FileImageInstance


class FileImageDataset(Dataset):
    """
    This dataset loads a dataset of images from a filesystem.
    The dataset should be stored on the filesystem as follows:

    <root dataset dir>
        fake
            <label 1_1fake>
            - image 1_1fake
            - image 2
        camera1
            <label 2>
            - image 1_1fake
            - image 2
        ...

    Non-image files are ignored.

    :param path: The path to the root folder of the dataset.
    :param name: The name of the dataset.
    :param return_binary: If True, the binary label is returned (real/fake) instead of the model name.
    :param split_file: The path to a file containing the filenames of the images that should be returned.
    """

    def __init__(self, path : str, name : str = None, split_file: str = None, return_binary: bool=False):
        super(FileImageDataset, self).__init__(name)
        self.path = path
        self.return_binary = return_binary

        # If split file provided store the filenames of included instances in a list
        if split_file:
            with open(split_file, 'r') as f:
                self.included_instances = f.read().splitlines()
        else:
            self.included_instances = None

    @cached_property
    def label_mapping(self):
        mapping = {}
        # Loop over folders (authenticity class) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                for subfolder in os.listdir(os.path.join(self.path, folder)):
                    # If directory
                    if os.path.isdir(os.path.join(self.path, folder, subfolder)):
                        mapping[subfolder] = folder
        return mapping

    def __iter__(self) -> Iterable[FileImageInstance]:
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
                                self.label_mapping[subfolder] = folder
                                if self.included_instances is None or img in self.included_instances:
                                    # If return binary labels
                                    if self.return_binary:
                                        yield FileImageInstance(os.path.join(self.path, folder, subfolder, img),
                                                                folder)
                                    else:
                                        yield FileImageInstance(os.path.join(self.path, folder, subfolder, img),
                                                                subfolder)
                            else:
                                print("Found file that is not a jpg, jpeg or png file: {}".format(img))