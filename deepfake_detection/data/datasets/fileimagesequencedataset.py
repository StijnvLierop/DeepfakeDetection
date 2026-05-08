import os
from pathlib import Path
from typing import List

from deepfake_detection.data.annotation import Annotation
from deepfake_detection.data.dataset import Dataset, MapStyleDatasetMixin
from deepfake_detection.data.instance import FileImageSequenceInstance, Instance


class FileImageSequenceDataset(MapStyleDatasetMixin, Dataset):
    """
    This dataset loads a dataset of images from a filesystem.
    The dataset should be stored on the filesystem as follows:

    <root dataset dir>
        real
            <label 1>
                - <image sequence 1>
                    - frame 1
                    - frame 2
                - <image sequence 2>
                    - frame 1
                    - frame 2
        fake
            <label 2>
                - <image sequence 1>
                    - frame 1
                    - frame 2
                - <image sequence 2>
                    - frame 1
                    - frame 2
        ...

    Non-image files are ignored.

    :param path: The path to the root folder of the dataset.
    :param dataset_name: The name of the dataset.
    :param split_file: The path to a file containing the directory names of the image sequences that should be returned.
    """

    def __init__(self, path: str, dataset_name: str = None, split_file: str = None):
        super().__init__(dataset_name)
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
        # Loop over folders (models) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                # Loop over folders (models) in dataset
                for subfolder in os.listdir(os.path.join(self.path, folder)):
                    # If directory
                    if os.path.isdir(os.path.join(self.path, folder, subfolder)):
                        # Loop over subdirs with image sequences inside them
                        for img_folder in os.listdir(
                            os.path.join(self.path, folder, subfolder)
                        ):
                            # if folder
                            if os.path.isdir(
                                os.path.join(self.path, folder, subfolder, img_folder)
                            ):
                                if (
                                    self.included_instances is None
                                    or img_folder in self.included_instances
                                ):
                                    paths.append(
                                        Path(
                                            os.path.join(
                                                self.path, folder, subfolder, img_folder
                                            )
                                        )
                                    )
        return paths

    def __getitem__(self, idx: int) -> Instance:
        # Get instance path
        path = self.instance_paths[idx]

        # Get labels from path
        authenticity_label, source_label, img_name = path.parts[-3:]

        # Return instance
        return FileImageSequenceInstance(
            str(path),
            Annotation({"authenticity": authenticity_label, "source": source_label}),
        )

    def __len__(self):
        return len(self.instance_paths)
