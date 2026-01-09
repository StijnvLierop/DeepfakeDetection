from typing import Optional, Mapping, Any, Union

import datasets
from datasets import load_dataset
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
        data_col: str,
        source_label_col: Optional[str] = None,
        authenticity_label_col: Optional[str] = None,
        authenticity_label_mapping: Optional[Mapping[Any, str]] = None,
        **kwargs,
    ):
        """
        :param dataset: Huggingface dataset object or ID of dataset.
        :param data_col: Name of the column containing the image data.
        :param source_label_col: Name of the column containing the source label.
        :param authenticity_label_col: Name of the column containing the authenticity label.
        :param authenticity_label_mapping: Mapping from labels to 'real', 'fake' or 'manipulated'.
                                           A wildcard '*' can be used to map all non-specified labels to the same label.
        :param **kwargs: Additional arguments passed to datasets.load_dataset.
        """
        super().__init__(
            name=dataset if isinstance(dataset, str) else dataset.config_name
        )
        self.data_col = data_col
        self.source_label_col = source_label_col
        self.authenticity_label_col = authenticity_label_col
        self.authenticity_label_mapping = authenticity_label_mapping

        # Load dataset if str provided
        if isinstance(dataset, str):
            self.dataset = load_dataset(dataset, **kwargs)
        # Otherwise assume dataset is already loaded
        else:
            self.dataset = dataset

        # Select the correct instance class
        data_type = self.dataset.features[self.data_col]
        if isinstance(data_type, Image):
            self.instance_class = ImageInstance
        else:
            raise ValueError(
                f"Loading data of type {data_type} "
                f"not supported with available instance classes."
            )

    def __getitem__(self, idx: int) -> ImageInstance:
        # Get sample
        sample = self.dataset[idx]

        # Create annotation (if present)
        annotation = None
        source_label = None
        authenticity_label = None
        if self.source_label_col:
            source_label = sample[self.source_label_col]

        if self.authenticity_label_col:
            authenticity_label = sample[self.authenticity_label_col]
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
        instance = self.instance_class(
            data=sample[self.data_col], annotation=annotation
        )

        return instance

    def __len__(self):
        return len(self.dataset)
