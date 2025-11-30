from typing import Optional, Mapping, Any

from datasets import load_dataset
from datasets.features import Image

from deepfake_detection.data import Dataset, ImageInstance
from deepfake_detection.data.annotation import Annotation


class HuggingfaceDataset(Dataset):
    """
    Class that can be used to load a Huggingface dataset as a deepfake_detection.data.Dataset.
    """

    def __init__(self,
                 repo_id: str,
                 data_col: str,
                 source_label_col: Optional[str] = None,
                 authenticity_label_col: Optional[str] = None,
                 authenticity_label_mapping: Optional[Mapping[Any, str]] = None,
                 **kwargs):
        """
        :param repo_id: Huggingface dataset ID.
        :param data_col: Name of the column containing the image data.
        :param source_label_col: Name of the column containing the source label.
        :param authenticity_label_col: Name of the column containing the authenticity label.
        :param authenticity_label_mapping: Mapping from labels to 'real', 'fake' or 'manipulated'.
        :param **kwargs: Additional arguments passed to datasets.load_dataset.
        """
        super().__init__(name=repo_id)
        self.repo_id = repo_id
        self.data_col = data_col
        self.source_label_col = source_label_col
        self.authenticity_label_col = authenticity_label_col
        self.authenticity_label_mapping = authenticity_label_mapping

        # Load dataset
        self.dataset = load_dataset(repo_id, **kwargs)

        # Select the correct instance class
        data_type = self.dataset.features[self.data_col]
        if isinstance(data_type, Image):
            self.instance_class = ImageInstance
        else:
            raise ValueError(f"Loading data of type {data_type} "
                             f"not supported with available instance classes.")


    def __iter__(self):
        # Loop over samples in dataset
        for sample in self.dataset:

            # Create annotation (if present)
            annotation = None
            source_label = None
            authenticity_label = None
            if self.source_label_col:
                source_label = sample[self.source_label_col]

            if self.authenticity_label_col:
                authenticity_label = sample[self.authenticity_label_col]
                if self.authenticity_label_mapping:
                    authenticity_label = self.authenticity_label_mapping[authenticity_label]

            if source_label is not None or authenticity_label is not None:
                annotation = Annotation(source_label=source_label,
                                        authenticity_label=authenticity_label
                                        )

            # Create instance
            instance = self.instance_class(data=sample[self.data_col], annotation=annotation)

            yield instance