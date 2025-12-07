import logging
import os
import re
from typing import Iterable

from deepfake_detection.data.annotation import Annotation
from deepfake_detection.data.dataset import Dataset
from deepfake_detection.data.instance import FileImageInstance


class GenImageDataset(Dataset):
    """
    This dataset loads a dataset of images from a filesystem given the structure of the GenImage dataset.
    The dataset should be stored on the filesystem as follows:

    <root dataset dir>
        <label 1_1fake>
            train
                ai
                    - image 1_1fake
                    - image 2
                natural
                    - image 1_1fake
                    - image 2
            val
                ...
        <label 2>
            ...
        ...

    Non-image files are ignored.

    :param path: The path to the root folder of the dataset.
    :param name: The name of the dataset.
    :param split: The split of the dataset to load. 'Train', 'val', or None to retrieve all data.
    """

    def __init__(self, path : str, split: str = None, name: str = None):
        super(GenImageDataset, self).__init__(name)
        self.path = path
        if split == 'train':
            self.split = ['train']
        elif split == 'val':
            self.split = ['val']
        else:
            self.split = ['train', 'val']


    def __iter__(self) -> Iterable[FileImageInstance]:
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
                                        yield FileImageInstance(
                                            os.path.join(self.path, generator, split, binary_label, img),
                                            Annotation(authenticity_label="real" if generator == "nature" else "fake",
                                                       source_label=self._format_label(generator)),
                                        )
                                    else:
                                        logging.debug("Found file that is not a jpg, jpeg or png file: {}".format(img))


    def _format_label(self, label: str) -> str:
        """
        Format the given label to a standard form so labels of different datasets can be compared.

        :param label: The label to format.
        :return: The formatted label.
        """

        # Convert to lowercase
        label = label.lower()

        # For stable diffusion labels, correctly format version number
        print(label)
        if label.startswith('stable diffusion'):
            label = re.sub(r'v_(\d)_(\d)', r'v$1.$2', label)
        print(label)
        # Replace any underscores with spaces
        label = label.replace('_', ' ')

        return label