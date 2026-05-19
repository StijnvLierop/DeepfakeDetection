from pathlib import Path
from typing import List, Optional

from deepfake_detection.data.annotation import Annotation
from deepfake_detection.data.dataset import Dataset, MapStyleDatasetMixin
from deepfake_detection.data.instance import Instance, FileImageInstance


class FileImageDataset(MapStyleDatasetMixin, Dataset):
    """
    This dataset loads a dataset of images from a filesystem.
    The following image files are supported: jpg, jpeg, png, bmp, webp, tiff.

    :param path: The path to the root folder of the dataset.
    :param glob_pattern: An optional glob pattern to filter the files that should be included in the dataset.
    :param dataset_name: The name of the dataset.
    :param labels: An optional list of labels equal to the depth of the folders.
                   Each label will be mapped to the values parsed from the folder names from shallow to deep.
    """

    def __init__(
        self,
        path: str,
        glob_pattern: Optional[str] = None,
        dataset_name: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ):
        super(FileImageDataset, self).__init__(dataset_name)
        self.path = path
        self.labels = labels
        self.glob_pattern = glob_pattern

        # Store instance paths
        self.instance_paths = self._index()

    def _index(self) -> List[Path]:
        """
        Indexes all files in the dataset and returns a list of filepaths.
        """
        # use recursive glob when no pattern provided
        glob_func = Path(self.path).glob if self.glob_pattern else Path(self.path).rglob

        # Gather all filepaths in the folder
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        image_paths = [
            Path(p)
            for p in glob_func(self.glob_pattern if self.glob_pattern else "*")
            if p.suffix.lower() in extensions
        ]
        return image_paths

    def __getitem__(self, idx: int) -> Instance:
        # Get instance path
        instance_path = self.instance_paths[idx]

        # Get all layers from root (label_values)
        label_values = Path(instance_path).relative_to(self.path).parts[:-1]

        # If labels are provided, ensure the nr is the same as label values
        if self.labels:
            if len(self.labels) < len(label_values):
                raise ValueError(
                    "Length of provided labels list is not equal or longer as the number of extracted label values!"
                    f"{self.labels} < {label_values}"
                )
        else:
            self.labels = range(len(label_values))

        # Return instance
        return FileImageInstance(
            str(instance_path),
            Annotation(
                labels={
                    str(label): str(value)
                    for label, value in zip(
                        self.labels[: len(label_values)], label_values
                    )
                }
            ),
        )

    def __len__(self):
        return len(self.instance_paths)
