import os
from typing import Iterable

from deepfake_detection.data.datasets.dataset import Dataset
from deepfake_detection.data.datasets.instance import VideoInstance


class FileVideoDataset(Dataset):
    """
    This dataset loads a dataset of videos from a filesystem.
    The dataset should be stored on the filesystem as follows:

    <root dataset dir>
        real
            <label 1>
                - video1
        fake
            <label 2>
                - video2
        ...

    Non-video files are ignored.

    :param path: The path to the root folder of the dataset.
    :param name: The name of the dataset.
    """

    def __init__(self, name: str, path: str):
        super().__init__(name)
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
                # Loop over classes (models) in dataset
                for subfolder in os.listdir(os.path.join(self.path, folder)):
                    # If directory
                    if os.path.isdir(os.path.join(self.path, folder, subfolder)):
                        # Loop over subdirs with videos inside them
                        for img_folder in os.listdir(os.path.join(self.path, folder, subfolder)):
                            # Loop over videos
                            for video in os.listdir(os.path.join(self.path, folder, subfolder)):
                                if video.split('.')[-1] in ['mp4', 'mov']:
                                    n += 1
        return n

    def __iter__(self) -> Iterable[VideoInstance]:
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
                            if video.split('.')[-1] in ['mp4', 'mov']:
                                yield VideoInstance(os.path.join(self.path, folder, subfolder),
                                                    subfolder,
                                                    folder)
                            else:
                                raise print("Found file that is not a mp4 or mov file: {}".format(video))