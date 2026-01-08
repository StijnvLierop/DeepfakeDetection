import json
import os
from typing import Iterable, Union

from deepfake_detection.data.annotation import Annotation
from deepfake_detection.data.dataset import Dataset
from deepfake_detection.data.instance import (
    FileVideoInstance,
    FileImageSequenceInstance,
)


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
    :param c_level: The compression level to return. Should be one of: 'c23', 'c40' or 'raw'. If none (default),
                    instances from all compression levels are returned.
    :param split: The split to return. Can be one of 'train', 'test' or 'validation'. If none (default), all instances
                  are returned.
    :param split_dict_path: A path to a JSON file containing a dictionary mapping of filename indices to splits.
    """

    def __init__(
        self,
        path: str,
        name: str = None,
        modality: str = "videos",
        c_level: str = None,
        split: str = None,
        split_dict_path: str = None,
    ):
        super().__init__(name)
        self.path = path

        # Ensure that a valid value is passed for modality
        if modality not in ["images", "videos"]:
            raise ValueError(
                f'Invalid modality: {modality}. Must be one of "images" or "videos".'
            )
        else:
            self.modality = modality

        # Ensure that a valid value is passed for c_level
        if c_level not in ["c23", "c40", "raw", None]:
            raise ValueError(
                f'Invalid c_level: {c_level}. Must be one of "c23", "c40", "raw" or None.'
            )
        if c_level is None:
            self.c_levels = ["c23", "c40", "raw"]
        else:
            self.c_levels = [c_level]

        # Ensure that a valid value is passed for split
        if split not in ["train", "test", "validation", None]:
            raise ValueError(
                f'Invalid split: {split}. Must be one of "train", "test", "validation" or None.'
            )
        else:
            # Set correct split
            self.split = split

        # Ensure that valid split mapping file is provided
        if split and split_dict_path is None:
            raise ValueError(
                f"Invalid split dict path: {split_dict_path}. Please provide a valid path to a "
                f".JSON file with a dictionary mapping of filename indices to splits."
            )

        # Load split dict file if provided
        if split_dict_path:
            with open(split_dict_path, "r") as f:
                self.split_dict = json.load(f)
        else:
            self.split_dict = None

    def __iter__(self) -> Iterable[Union[FileImageSequenceInstance, FileVideoInstance]]:
        # Loop over folders (authenticity class) in dataset
        for folder in os.listdir(self.path):
            # Loop over folders (models) in dataset
            for subfolder in os.listdir(os.path.join(self.path, folder)):
                # Loop over qualities (compression levels) in dataset
                for c_level in self.c_levels:
                    # Loop over instances
                    for instance in os.listdir(
                        os.path.join(
                            self.path, folder, subfolder, c_level, self.modality
                        )
                    ):
                        # If instance in split
                        if self._instance_in_split(instance):
                            if self.modality == "images":
                                yield FileImageSequenceInstance(
                                    os.path.join(
                                        self.path,
                                        folder,
                                        subfolder,
                                        c_level,
                                        self.modality,
                                        instance,
                                    ),
                                    Annotation(
                                        authenticity_label=folder,
                                        source_label=subfolder,
                                    ),
                                )
                            else:
                                yield FileVideoInstance(
                                    os.path.join(
                                        self.path,
                                        folder,
                                        subfolder,
                                        c_level,
                                        self.modality,
                                        instance,
                                    ),
                                    Annotation(
                                        authenticity_label=folder,
                                        source_label=subfolder,
                                    ),
                                )

    def _instance_in_split(self, instance_path: str) -> bool:
        """
        This functions returns true if the current instance is in the selected split. If no split is selected or no
        split mapping is provided, it will always return true.

        :param instance_path: The path to the instance.
        """
        # If no split dict is set or split is set to None return true
        if self.split is None or self.split_dict is None:
            return True

        # Extract filename index
        filename_index = instance_path.split(".")[0]

        # Check if instance is in current split
        if (
            filename_index in self.split_dict.keys()
            and self.split_dict[filename_index] == self.split
        ):
            return True
        else:
            return False
