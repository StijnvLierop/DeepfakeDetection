import os
from typing import Iterable

from deepfake_detection.data.datasets.dataset import Dataset
from deepfake_detection.data.datasets.instance import ImageInstance


class GenImageDataset(Dataset):
    """
    This dataset loads a dataset of images from a filesystem given the structure of the GenImage dataset.
    The dataset should be stored on the filesystem as follows:

    <root dataset dir>
        <label 1>
            train
                ai
                    - image 1
                    - image 2
                natural
                    - image 1
                    - image 2
            val
                ...
        <label 2>
            ...
        ...

    Non-image files are ignored.

    :param path: The path to the root folder of the dataset.
    :param name: The name of the dataset.
    :param split: The split of the dataset to load. 'train', 'val', or None to retrieve all data.
    """

    def __init__(self, path : str, name : str, split: str = None):
        super(GenImageDataset, self).__init__(name=name)
        self.path = path
        if split == 'train':
            self.split = ['train']
        elif split == 'val':
            self.split = ['val']
        else:
            self.split = ['train', 'val']

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        n = 0
        # Loop over generators
        for generator in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, generator)):
                for split in self.split:
                    # If directory
                    if os.path.isdir(os.path.join(self.path, generator, split)):
                        # Loop over folders (label)
                        for binary_label in os.listdir(os.path.join(self.path, generator, split)):
                            # If directory
                            if os.path.isdir(os.path.join(self.path, generator, split, binary_label)):
                                # Loop over images
                                for img in os.listdir(os.path.join(self.path, generator, split, binary_label)):
                                    if img.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
                                        n += 1
        return n

    def __iter__(self) -> Iterable[ImageInstance]:
        # Loop over generators
        for generator in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, generator)):
                for split in self.split:
                    # If directory
                    if os.path.isdir(os.path.join(self.path, generator, split)):
                        # Loop over folders (label)
                        for binary_label in os.listdir(os.path.join(self.path, generator, split)):
                            # If directory
                            if os.path.isdir(os.path.join(self.path, generator, split, binary_label)):
                                # Loop over images
                                for img in os.listdir(os.path.join(self.path, generator, split, binary_label)):
                                    if img.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
                                        yield ImageInstance(
                                            os.path.join(self.path, generator, split, binary_label, img),
                                            class_label=generator if binary_label == 'ai' else 'real',
                                            authenticity_label='fake' if binary_label == 'ai' else 'real',
                                        )
                                    else:
                                        print("Found file that is not a jpg, jpeg or png file: {}".format(img))