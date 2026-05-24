import os
from pathlib import Path
from typing import Optional, Sequence
from itertools import zip_longest

import numpy as np
from PIL import Image
import fiftyone as fo
import fiftyone.brain as fob
from fiftyone.utils.data.importers import GenericSampleDatasetImporter

from deepfake_detection.analysis.transforms.base import AnalysisTransform
from deepfake_detection.data.instance import Instance
from deepfake_detection.data.dataset import Dataset
from deepfake_detection.models import Prediction, Model

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


class FiftyOneDatasetImporter(GenericSampleDatasetImporter):
    """
    Helper class that is used to load Dataset samples as FiftyOne samples.
    """

    def __init__(
        self,
        dataset: Dataset,
        predictions: Optional[Sequence[Prediction]] = None,
        cache_dir: Optional[str] = None,
        transforms: Sequence[AnalysisTransform] = (),
    ):
        """
        :param dataset: The deepfake_detection.data.Dataset to import.
        :param predictions: An optional sequence of predictions.
        :param cache_dir: The directory to store temporary files in case
                          instances are not stored on disk yet.
        :param transforms: Analysis transforms whose results are attached to each
                           sample as named heatmap fields.
        """
        super().__init__(dataset_dir=None, shuffle=False, seed=None, max_samples=None)
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.predictions = predictions
        self.transforms = transforms

    @property
    def has_sample_field_schema(self):
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
        for instance, prediction in zip_longest(
            self.dataset, self.predictions or [], fillvalue=None
        ):
            yield _instance_to_fo_sample(instance, prediction, self.cache_dir, self.transforms)


def _array_to_fo_heatmap(array: np.ndarray) -> fo.Heatmap:
    """Convert an analysis result array to a FiftyOne Heatmap field."""
    if array.ndim == 3:
        array = array.max(axis=-1)
    array = array.astype(np.float32)
    lo, hi = array.min(), array.max()
    if hi > lo:
        array = (array - lo) / (hi - lo)
    else:
        array = np.zeros_like(array)
    return fo.Heatmap(map=array)


def _instance_to_fo_sample(
    instance: Instance,
    prediction: Optional["Prediction"],
    cache_dir: Optional[str],
    transforms: Sequence[AnalysisTransform] = (),
) -> fo.Sample:
    """
    Converts a given instance to a FiftyOne sample.

    :param instance: The instance to convert.
    :param prediction: Optional prediction for the instance.
    :param cache_dir: Optional directory to cache instances if they don't have a path.
    """
    # If instance has an image path
    if (
        hasattr(instance, "path")
        and os.path.splitext(instance.path)[1].lower() in IMAGE_EXTS
    ):
        path = instance.path

    # Otherwise export the file to a temporary directory
    # and use that path
    else:
        # Raise ValueError in case cache_dir is not set
        if cache_dir is None:
            raise ValueError(
                "cache_dir must be set when converting "
                "a dataset that does not contain pointers"
                " to files on disk."
            )

        # Save instance to temporary directory
        path = Path(os.path.join(cache_dir, str(instance.__hash__())))
        path = instance.save(path)

    # Create sample
    sample = fo.Sample(filepath=path)

    # Add annotations
    if instance.annotation:
        # Add labels
        for label in instance.annotation.labels:
            sample[label] = fo.Classification(
                label=instance.annotation.get_label(label)
            )
        # Add mask
        if instance.annotation.mask is not None:
            # Convert mask to numpy array
            mask_array = np.array(instance.annotation.mask.convert("L"), dtype=np.uint8)
            # Drop any trailing channel dimensions (e.g., turning (H, W, 1) -> (H, W))
            if len(mask_array.shape) == 3:
                if mask_array.shape[2] == 1:
                    mask_array = np.squeeze(mask_array, axis=2)
            elif len(mask_array.shape) != 2:
                raise ValueError(
                    f"Expected a 2D mask array, but got shape {mask_array.shape}"
                )
            sample["mask"] = fo.Segmentation(mask=mask_array)

    # Add predictions
    if prediction:
        # Add all predicted labels
        predictions = [
            fo.Classification(
                label=label, confidence=prediction.classification.get(label)
            )
            for label in prediction.classification
        ]
        sample["predictions"] = fo.Classifications(classifications=predictions)

        # Add the predicted label with the highest confidence score
        top_pred = max(prediction.classification, key=prediction.classification.get)
        sample["predicted_label"] = fo.Classification(
            label=top_pred, confidence=prediction.classification[top_pred]
        )

    # Apply analysis transforms and attach results as heatmap fields
    if transforms and hasattr(instance, "data") and isinstance(instance.data, Image.Image):
        img = np.array(instance.data)
        for transform in transforms:
            result = transform.apply(img)
            sample[transform.name] = _array_to_fo_heatmap(result)

    return sample


def to_fiftyone_dataset(
    dataset: Dataset,
    batch_size: int = 128,
    predictions: Optional[Sequence[Prediction]] = None,
    cache_dir: Optional[str] = None,
    embedding_model: Optional[Model] = None,
    fo_dataset: Optional[fo.Dataset] = None,
    transforms: Sequence[AnalysisTransform] = (),
) -> fo.Dataset:
    """
    Function that converts a Dataset to a FiftyOne dataset.

    :param dataset: The dataset to convert.
    :param batch_size: The batch size for computing embeddings.
    :param predictions: An optional sequence of predictions.
    :param cache_dir: The directory to store temporary files in case
                      instances are not stored on disk yet.
    :param embedding_model: The model to use for computing embeddings (optional).
    :param fo_dataset: An existing FiftyOne dataset to populate. When provided,
                       the caller is responsible for prior deletion/creation so
                       the app can be launched before samples start loading.
    :param transforms: Analysis transforms whose results are attached to each
                       sample as named heatmap fields.
    :return: The populated FiftyOne dataset.
    """
    # If no fo_dataset is provided, create a new one
    if fo_dataset is None:
        if fo.dataset_exists(dataset.dataset_name):
            fo.delete_dataset(dataset.dataset_name)
        fo_dataset = fo.Dataset(name=dataset.dataset_name)

    # Convert predictions to list if provided
    predictions_list = list(predictions) if predictions else []

    # Calculate embeddings when an embedding model is provided
    if embedding_model:
        # Stream samples and compute embeddings batch-by-batch so the app
        # shows results progressively instead of blocking until all done.
        embeddings = []
        pred_idx = 0

        for batch_instances in dataset.iter(batch_size=batch_size):
            # Calculate predictions for embedding model
            batch_embeds = embedding_model.predict_batch(batch_instances)
            if batch_embeds[0].embedding is None:
                raise ValueError("Provided model does not return embeddings!")
            embeddings.extend([e.embedding for e in batch_embeds])

            samples = []
            for instance in batch_instances:
                pred = (
                    predictions_list[pred_idx]
                    if pred_idx < len(predictions_list)
                    else None
                )
                samples.append(_instance_to_fo_sample(instance, pred, cache_dir, transforms))
                pred_idx += 1
            fo_dataset.add_samples(samples)

        fob.compute_visualization(
            fo_dataset,
            brain_key=embedding_model.name,
            embeddings=embeddings,
            num_dims=2,
            method="umap",
            seed=42,
        )
    else:
        # Stream samples in batches so a live session updates progressively
        importer = FiftyOneDatasetImporter(dataset, predictions, cache_dir, transforms)
        batch = []
        for sample in importer:
            batch.append(sample)
            if len(batch) >= batch_size:
                fo_dataset.add_samples(batch)
                batch.clear()
        if batch:
            fo_dataset.add_samples(batch)

    return fo_dataset
