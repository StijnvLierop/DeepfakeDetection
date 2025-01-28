import os
from typing import Iterable

from deepfake_detection.data.datasets.dataset import Dataset
from deepfake_detection.data.datasets.instance import VideoInstance

class FileVideoDataset(Dataset):

    def __init__(self, path):
        self.path = path

    def __iter__(self) -> Iterable[VideoInstance]:
        # Loop over folders (models) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                # Loop over videos
                for video in os.listdir(os.path.join(self.path, folder)):
                    yield VideoInstance(os.path.join(self.path, folder, video), folder)