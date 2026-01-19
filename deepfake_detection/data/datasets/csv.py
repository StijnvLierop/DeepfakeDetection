import os
from typing import Optional

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
                 data_col: str,
                 authenticity_label_col: str,
                 source_label_col: Optional[str] = None,
                 authenticity_label_mapping: Optional[dict] = None,
                 dataset_name: Optional[str] = None):
        """
        :param csv_path: Path to the CSV file.
        :param data_folder: Path to the folder containing the samples.
        :param data_col: Name of the column containing the sample path.
        :param authenticity_label_col: Name of the column containing the authenticity label.
        :param source_label_col: Name of the column containing the source label (optional).
        :param authenticity_label_mapping: Mapping from labels to labels.
        """
        super().__init__(name=dataset_name)
        self.csv_path = csv_path
        self.data_folder = data_folder
        self.data_col = data_col
        self.source_label_col = source_label_col
        self.authenticity_label_col = authenticity_label_col
        self.authenticity_label_mapping = authenticity_label_mapping

        # Read csv file
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Instance:
        row = self.data.iloc[idx]

        # Get instance
        sample_path = os.path.join(self.data_folder, row[self.data_col])
        if row[self.data_col].split('.')[-1] in ('jpg', 'jpeg', 'png', 'webp'):
            instance_class = FileImageInstance
        else:
            raise ValueError("Unknown filetype: " + row[self.data_col])

        # Create annotation (if present)
        annotation = None
        source_label = None
        authenticity_label = None
        if self.source_label_col:
            source_label = row[self.source_label_col]

        if self.authenticity_label_col:
            authenticity_label = row[self.authenticity_label_col]
            if self.authenticity_label_mapping:
                # Check if label is in mapping and map if so
                if authenticity_label in self.authenticity_label_mapping:
                    authenticity_label = self.authenticity_label_mapping[authenticity_label]
                # Otherwise check for wildcard
                elif "*" in self.authenticity_label_mapping:
                    authenticity_label = self.authenticity_label_mapping["*"]
                # Otherwise raise error
                else:
                    raise ValueError(f"Label '{authenticity_label}' not found in authenticity_label_mapping"
                                     f" and no wildcard '*' is provided.")

        if source_label is not None or authenticity_label is not None:
            annotation = Annotation(
                source_label=source_label, authenticity_label=authenticity_label
            )

        # Create instance
        return instance_class(sample_path, annotation=annotation)
