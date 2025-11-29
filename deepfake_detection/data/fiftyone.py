import os
import tempfile
from typing import Optional

import fiftyone as fo
from fiftyone.utils.data.importers import GenericSampleDatasetImporter

from deepfake_detection.data import Instance, ImageInstance
from deepfake_detection.data.dataset import Dataset


class FiftyOneDatasetImporter(GenericSampleDatasetImporter ):
    """
    Helper class that is used to load Dataset samples as FiftyOne samples.
    """
    def __init__(self, dataset: Dataset, cache_dir: Optional[str] = None):
        """
        :param dataset: The deepfake_detection.data.Dataset to import.
        :param cache_dir: The directory to store temporary files in case
                          instances are not stored on disk yet.
        """
        super().__init__(dataset_dir=None,
                         shuffle=False,
                         seed=None,
                         max_samples=None)
        self.dataset = dataset
        self.cache_dir = cache_dir


    @property
    def has_sample_field_schema(self):
        """Whether this importer produces a sample field schema."""
        return False


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
        for instance in self.dataset:

            # If an instance path is available, set path
            if hasattr(instance, 'path'):
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
                path = save_to_cache_dir(instance, self.cache_dir)

            # Create sample
            sample = fo.Sample(path)

            # Add annotations
            sample['source'] = fo.Classification(label=instance.annotation.source_label)
            sample['authenticity'] = fo.Classification(label=instance.annotation.authenticity_label)

            yield sample


def to_fiftyone_dataset(dataset: Dataset,
                        cache_dir: Optional[str] = None) -> fo.Dataset:
    """
    Function that converts a Dataset to a FiftyOne dataset.

    :param dataset: The dataset to convert.
    :param cache_dir: The directory to store temporary files in case
                      instances are not stored on disk yet.
    :return: The converted FiftyOne dataset.
    """

    # Create dataset importer object
    dataset_importer = FiftyOneDatasetImporter(dataset, cache_dir)

    # Create fiftyone dataset object from importer
    return fo.Dataset.from_importer(dataset_importer)


def save_to_cache_dir(instance: Instance, cache_dir: str) -> str:
    """
    Helper function that saves sample data to a temp file and returns its path.

    :param instance: The instance to save to a temporary directory.
    :param cache_dir: The directory to store temporary files in case
                      instances are not stored on disk yet.
    :return: The path to the saved file.
    """
    # Create cache_dir if not yet exists
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    # If image instance
    if isinstance(instance, ImageInstance):
        path = os.path.join(cache_dir, str(instance.__hash__()) + '.png')
        instance.data.save(path, 'PNG')

    else:
        raise ValueError(f"Unsupported instance type: {type(instance)}")

    return path