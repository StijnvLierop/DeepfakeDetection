import os
from typing import Optional, Mapping, Any, Union, List

import datasets
from datasets import load_dataset, load_from_disk
from datasets.features import Image

from deepfake_detection.data.dataset import Dataset, MapStyleDatasetMixin
from deepfake_detection.data.instance import ImageInstance
from deepfake_detection.data.annotation import Annotation


class HuggingfaceDataset(MapStyleDatasetMixin, Dataset):
    """
    Class that can be used to load a Huggingface dataset as a deepfake_detection.data.Dataset.
    """

    def __init__(
        self,
        dataset: Union[str, datasets.Dataset],
        instance_col: str,
        label_cols: Optional[Mapping[str, str]] = {},
        dataset_name: Optional[str] = None,
        **kwargs,
    ):
        """
        :param dataset: Huggingface dataset object or ID of dataset.
        :param instance_col: Name of the column containing the image data.
        :param dataset_name: Name of the dataset.
        :param label_cols: A mapping from column names to label keys. Each column will be mapped
                           to the corresponding label in 'Annotation'.
        :param **kwargs: Additional arguments passed to datasets.load_dataset.
        """
        super().__init__(dataset_name=dataset_name)
        self.instance_col = instance_col
        self.label_cols = label_cols

        # If no string is provided, assume a dataset is already loaded
        if not isinstance(dataset, str):
            self.dataset = dataset
        else:
            # Check if the dataset exists locally and use the load_from_disk function
            if os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, "dataset_info.json")):
                self.dataset = load_from_disk(dataset, **kwargs)
            # Otherwise use load_dataset
            else:
                self.dataset = load_dataset(dataset, **kwargs)

        # Select the correct instance class
        data_type = self.dataset.features[self.instance_col]
        if isinstance(data_type, Image):
            self.instance_class = ImageInstance
        else:
            raise ValueError(
                f"Loading data of type {data_type} "
                f"not supported with available instance classes."
            )

    def __getitem__(self, idx):

        if isinstance(idx, slice):
            instances = []
            for i in range(len(self.dataset[idx])):
                instances.append(self.__getitem__(i))
            return instances

        # Get sample
        sample = self.dataset[idx]

        # Loop over label_cols
        labels = {}
        for col, label in self.label_cols.items():

            # Assign column value to label
            labels[label] = sample[col]

        # If labels found, add annotation object
        if labels != {}:
            annotation = Annotation(labels=labels)
        else:
            annotation = None

        # Create instance
        instance = self.instance_class(
            data=sample[self.instance_col], annotation=annotation
        )
        return instance

    def __len__(self):
        return len(self.dataset)
