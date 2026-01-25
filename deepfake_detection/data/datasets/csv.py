import os
from typing import Optional, Mapping

import pandas as pd

from deepfake_detection.data.annotation import Annotation
from deepfake_detection.data.dataset import MapStyleDatasetMixin, Dataset
from deepfake_detection.data.instance import Instance, FileImageInstance


class CSVDataset(MapStyleDatasetMixin, Dataset):
    """
    Dataset class that is loaded from a CSV file.
    """

    def __init__(self,
                 csv_path: str,
                 data_folder: str,
                 instance_col: str,
                 label_cols: Optional[Mapping[str, str]] = {},
                 dataset_name: Optional[str] = None):
        """
        :param csv_path: Path to the CSV file.
        :param data_folder: Path to the folder containing the samples.
        :param instance_col: Name of the column containing the sample path.
        :param label_cols: A mapping from column names to label keys. Each column will be mapped
                           to the corresponding label in 'Annotation'.
        :param dataset_name: Name of the dataset.
        """
        super().__init__(dataset_name=dataset_name)
        self.csv_path = csv_path
        self.data_folder = data_folder
        self.instance_col = instance_col
        self.label_cols = label_cols

        # Read csv file
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Instance:
        if isinstance(idx, slice):
            instances = []
            # Calculate range for slice
            for i in range(*idx.indices(len(self))):
                instances.append(self.__getitem__(i))
            return instances

        # Get row
        row = self.data.iloc[idx]

        # Get instance
        sample_path = os.path.join(self.data_folder, row[self.instance_col])
        if row[self.instance_col].split('.')[-1] in ('jpg', 'jpeg', 'png', 'webp'):
            instance_class = FileImageInstance
        else:
            raise ValueError("Unknown filetype: " + row[self.instance_col])

        # Loop over label_cols
        labels = {}
        for col, label in self.label_cols.items():
            # Assign column value to label
            labels[label] = row[col]

        # If labels found, add annotation object
        if labels != {}:
            annotation = Annotation(labels=labels)
        else:
            annotation = None

        # Create instance
        instance = instance_class(sample_path, annotation=annotation)
        return instance
