import os
from typing import Iterable

from deepfake_detection.data.datasets.dataset import Dataset
from deepfake_detection.data.datasets.instance import VideoInstance, ImageSequenceInstance


class FaceForensicsDataset(Dataset):
    """
    This dataset loads the FaceForensics++ dataset from a filesystem. The structure of the dataset should
    be the same as defined by the authors:

    <root dataset dir>
    |-- original_sequences
        |-- youtube
            < c0/raw original sequence images/videos of the FaceForensics++ dataset >
            < c23/hq original sequence images/videos >
            < c40/lq original sequence images/videos >
        |-- actors
            ...
    |-- manipulated_sequences
        |-- Deepfakes
            < images/videos ... as well as masks >
        |-- DeepFakeDetection
            < images/videos ... as well as masks >
        |-- Face2Face
            < images/videos ... as well as masks >
        |-- FaceSwap
            < images/videos ... as well as masks >
        |-- NeuralTextures
            < images/videos ... well as masks >

    The original dataset can be found here: https://github.com/ondyari/FaceForensics

    :param path: The path to the root folder of the dataset.
    :param name: The name of the dataset.
    :param modality: The modality to return. Can be one of 'images' or 'videos' (default).
                     When 'images' is selected, the dataset returns a series of ImageSequenceInstance. When 'videos'
                     is selected, the dataset returns a series of VideoInstance.
    """

    def __init__(self, name: str, path: str, modality: str = 'videos'):
        super().__init__(name)
        self.path = path

        # Ensure that a valid value is passed for modality
        if modality not in ['images', 'videos']:
            raise ValueError(f'Invalid modality: {modality}. Must be one of "images" or "videos".')
        else:
            self.modality = modality

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        n = 0
        # Loop over folders (authenticity class) in dataset
        for folder in os.listdir(self.path):
            # Loop over folders (models) in dataset
            for subfolder in os.listdir(os.path.join(self.path, folder)):
                # Loop over qualities (compression levels) in dataset
                    for c_level in ['raw', 'c23', 'c40']:
                        # Loop over instances
                        for _ in os.listdir(os.path.join(self.path, folder, subfolder, c_level, self.modality)):
                            n += 1
        return n

    @property
    def label_mapping(self):
        mapping = {}
        # Loop over folders (authenticity class) in dataset
        # Loop over folders (authenticity class) in dataset
        for folder in os.listdir(self.path):
            # Loop over folders (models) in dataset
            for subfolder in os.listdir(os.path.join(self.path, folder)):
                # Loop over qualities (compression levels) in dataset
                mapping[subfolder] = folder

        return mapping

    def __iter__(self) -> Iterable[VideoInstance]:
        # Loop over folders (authenticity class) in dataset
        for folder in os.listdir(self.path):
            # Loop over folders (models) in dataset
            for subfolder in os.listdir(os.path.join(self.path, folder)):
                # Loop over qualities (compression levels) in dataset
                for c_level in ['raw', 'c23', 'c40']:
                    # Loop over instances
                    for instance in os.listdir(os.path.join(self.path, folder, subfolder, c_level, self.modality)):
                        if self.modality == 'images':
                            yield ImageSequenceInstance(
                                os.path.join(self.path, folder, subfolder, c_level, self.modality, instance),
                                subfolder)
                        else:
                            yield VideoInstance(
                                os.path.join(self.path, folder, subfolder, c_level, self.modality, instance),
                                subfolder)