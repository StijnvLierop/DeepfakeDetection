import os
from pathlib import Path
from typing import Optional, Sequence
from itertools import zip_longest

import fiftyone as fo
from fiftyone.utils.data.importers import GenericSampleDatasetImporter
from fiftyone.core.fields import EmbeddedDocumentField

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.models import Prediction

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}


class FiftyOneDatasetImporter(GenericSampleDatasetImporter):
    """
    Helper class that is used to load Dataset samples as FiftyOne samples.
    """

    def __init__(self,
                 dataset: Dataset,
                 predictions: Optional[Sequence[Prediction]] = None,
                 cache_dir: Optional[Path] = None):
        """
        :param dataset: The deepfake_detection.data.Dataset to import.
        :param predictions: An optional sequence of predictions.
        :param cache_dir: The directory to store temporary files in case
                          instances are not stored on disk yet.
        """
        super().__init__(dataset_dir=None,
                         shuffle=False,
                         seed=None,
                         max_samples=None)
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.predictions = predictions

    @property
    def has_sample_field_schema(self):
        return True

    def get_sample_field_schema(self):
        schema = {
            "source_label": EmbeddedDocumentField(document_type=fo.Classification),
            "authenticity_label": EmbeddedDocumentField(document_type=fo.Classification),
        }
        if self.predictions:
            schema["predictions"] = EmbeddedDocumentField(document_type=fo.Classifications)
            schema["predicted_label"] = EmbeddedDocumentField(document_type=fo.Classification)
        return schema

    @property
    def has_dataset_info(self):
        """Whether this importer produces dataset info."""
        return False

    def setup(self):
        """Performs any necessary setup before importing."""
        pass

    def close(self, *args):
        """Performs any necessary cleanup after importing."""
        pass

    def __iter__(self):
        # Loop over instances in the dataset
        for instance, prediction in zip_longest(self.dataset, self.predictions or [], fillvalue=None):

            # If instance has an image path
            if (hasattr(instance, 'path') and
                    os.path.splitext(instance.path)[1].lower() in IMAGE_EXTS):
                path = instance.path

            # Otherwise export the file to a temporary directory
            # and use that path
            else:
                # Raise ValueError in case cache_dir is not set
                if self.cache_dir is None:
                    raise ValueError("cache_dir must be set when converting "
                                     "a dataset that does not contain pointers"
                                     " to files on disk.")

                # Save instance to temporary directory
                path = Path(os.path.join(self.cache_dir, str(instance.__hash__())))
                path = instance.save(path)

            # Create sample
            sample = fo.Sample(filepath=path)

            # Add annotations
            if instance.annotation:
                sample['source_label'] = fo.Classification(label=instance.annotation.source_label)
                sample['authenticity_label'] = fo.Classification(label=instance.annotation.authenticity_label)

            # Add predictions
            if prediction:
                # Add all predicted labels
                predictions = [fo.Classification(label=l,
                                                 confidence=prediction.classification.get(l))
                               for l in prediction.classification]
                sample['predictions'] = fo.Classifications(classifications=predictions)

                # Add the predicted label with the highest confidence score
                top_pred = max(prediction.classification,
                               key=prediction.classification.get)
                sample['predicted_label'] = fo.Classification(
                    label=top_pred,
                    confidence=prediction.classification[top_pred]
                )

            yield sample


def to_fiftyone_dataset(dataset: Dataset,
                        predictions: Optional[Sequence[Prediction]] = None,
                        cache_dir: Optional[Path] = None) -> fo.Dataset:
    """
    Function that converts a Dataset to a FiftyOne dataset.

    :param dataset: The dataset to convert.
    :param predictions: An optional sequence of predictions.
    :param cache_dir: The directory to store temporary files in case
                      instances are not stored on disk yet.
    :return: The converted FiftyOne dataset.
    """

    # Create dataset importer object
    dataset_importer = FiftyOneDatasetImporter(dataset, predictions, cache_dir)

    # Create fiftyone dataset object from importer
    return fo.Dataset.from_importer(dataset_importer)
