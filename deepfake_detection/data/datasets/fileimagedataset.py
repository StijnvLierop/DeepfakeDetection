import logging
import os
from pathlib import Path
from typing import List

from deepfake_detection.data.annotation import Annotation
from deepfake_detection.data.dataset import Dataset, MapStyleDatasetMixin
from deepfake_detection.data.instance import Instance, FileImageInstance


class FileImageDataset(MapStyleDatasetMixin, Dataset):
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
    :param split_file: The path to a file containing the filenames of the images that should be returned.
    """

    def __init__(self, path: str, name: str = None, split_file: str = None):
        super(FileImageDataset, self).__init__(name)
        self.path = path

        # If split file provided store the filenames of included instances in a list
        if split_file:
            with open(split_file, "r") as f:
                self.included_instances = f.read().splitlines()
        else:
            self.included_instances = None

        # Store instance paths
        self.instance_paths = self._index()

    def _index(self) -> List[Path]:
        """
        Indexes all files in the dataset and returns a list of filepaths.
        """
        # Loop over folders (labels) in dataset
        paths = []
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                for subfolder in os.listdir(os.path.join(self.path, folder)):
                    # If directory
                    if os.path.isdir(os.path.join(self.path, folder, subfolder)):
                        # Loop over images
                        for img in os.listdir(
                            os.path.join(self.path, folder, subfolder)
                        ):
                            if img.split(".")[-1].lower() in ["jpg", "jpeg", "png"]:
                                if (
                                    self.included_instances is None
                                    or img in self.included_instances
                                ):
                                    paths.append(
                                        Path(
                                            os.path.join(
                                                self.path, folder, subfolder, img
                                            )
                                        )
                                    )
                            else:
                                logging.debug(
                                    "Found file that is not a jpg, jpeg or png file: {}".format(
                                        img
                                    )
                                )
        return paths

    def __getitem__(self, idx: int) -> Instance:
        # Get instance path
        path = self.instance_paths[idx]

        # Get labels from path
        authenticity_label, source_label, img_name = path.parts[-3:]

        # Return instance
        return FileImageInstance(
            str(path),
            Annotation(
                authenticity_label=authenticity_label, source_label=source_label
            ),
        )

    def __len__(self):
        return len(self.instance_paths)
