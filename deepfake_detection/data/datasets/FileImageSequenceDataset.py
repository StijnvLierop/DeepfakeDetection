import os
from typing import Iterable

from deepfake_detection.data.datasets.dataset import Dataset
from deepfake_detection.data.datasets.instance import ImageSequenceInstance

class FileImageSequenceDataset(Dataset):

    def __init__(self, path):
        self.path = path

    def __iter__(self) -> Iterable[ImageSequenceInstance]:
        # Loop over folders (models) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                # Loop over images
                for img_folder in os.listdir(os.path.join(self.path, folder)):
                    # if folder
                    if os.path.isdir(os.path.join(self.path, folder, img_folder)):
                        yield ImageSequenceInstance(os.path.join(self.path, folder, img_folder), folder)