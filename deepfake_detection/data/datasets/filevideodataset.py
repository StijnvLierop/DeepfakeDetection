import logging
import os
from typing import Iterable

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.data.instance import FileVideoInstance

class FileVideoDataset(Dataset):
    """
    This dataset loads a dataset of videos from a filesystem.
    The dataset should be stored on the filesystem as follows:

    <root dataset dir>
        real
            <label 1_1fake>
                - video1
        fake
            <label 2>
                - video2
        ...

    Non-video files are ignored.

    :param path: The path to the root folder of the dataset.
    :param name: The name of the dataset.
    :param split_file: The path to a file containing the filenames of the videos that should be returned.
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

    @property
    def label_mapping(self):
        mapping = {}
        # Loop over folders (authenticity class) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                # Loop over folders (models) in dataset
                for subfolder in os.listdir(os.path.join(self.path, folder)):
                    # If directory
                    if os.path.isdir(os.path.join(self.path, folder, subfolder)):
                        # Loop over videos
                        for video in os.listdir(os.path.join(self.path, folder, subfolder)):
                            mapping[subfolder] = folder
        return mapping

    def __iter__(self) -> Iterable[FileVideoInstance]:
        # Loop over folders (authenticity class) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                # Loop over folders (models) in dataset
                for subfolder in os.listdir(os.path.join(self.path, folder)):
                    # If directory
                    if os.path.isdir(os.path.join(self.path, folder, subfolder)):
                        # Loop over videos
                        for video in os.listdir(os.path.join(self.path, folder, subfolder)):
                            if video.split('.')[-1].lower() in ['mp4', 'mov']:
                                if self.included_instances is None or video in self.included_instances:
                                    yield FileVideoInstance(os.path.join(self.path, folder, subfolder, video),
                                                            subfolder)
                            else:
                                raise logging.debug("Found file that is not a mp4 or mov file: {}".format(video))