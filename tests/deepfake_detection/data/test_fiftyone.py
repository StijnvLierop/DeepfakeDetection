import pytest

from deepfake_detection.data.instance import ImageInstance
from deepfake_detection.data.datasets import FileImageDataset, ListDataset
from deepfake_detection.data.fiftyone import to_fiftyone_dataset
from fiftyone.core.dataset import Dataset as FoDataset

from deepfake_detection.models import Prediction


def test_to_fiftyone_dataset_paths_available_without_predictions(image_dataset_path):
    image_dataset = FileImageDataset(image_dataset_path)
    converted_dataset = to_fiftyone_dataset(image_dataset)
    assert isinstance(converted_dataset, FoDataset)
    assert len(image_dataset) == len(converted_dataset)


def test_to_fiftyone_dataset_paths_available_with_predictions(image_dataset_path):
    image_dataset = FileImageDataset(image_dataset_path)
    predictions = [Prediction(classification={"test": 0.5}) for _ in image_dataset]
    converted_dataset = to_fiftyone_dataset(image_dataset, predictions)
    assert isinstance(converted_dataset, FoDataset)
    assert len(image_dataset) == len(converted_dataset)


def test_to_fiftyone_dataset_paths_unavailable_cache_dir(image_dataset_path, tmp_path):
    image_dataset = FileImageDataset(image_dataset_path)
    instances = [ImageInstance(data=i.data) for i in image_dataset]
    image_dataset = ListDataset(instances)
    converted_dataset = to_fiftyone_dataset(image_dataset, cache_dir=tmp_path)
    assert isinstance(converted_dataset, FoDataset)
    assert len(image_dataset) == len(converted_dataset)


def test_to_fiftyone_dataset_paths_unavailable_no_cache_dir(image_dataset_path):
    image_dataset = FileImageDataset(image_dataset_path)
    instances = [ImageInstance(data=i.data) for i in image_dataset]
    image_dataset = ListDataset(instances)
    with pytest.raises(ValueError):
        to_fiftyone_dataset(image_dataset)
