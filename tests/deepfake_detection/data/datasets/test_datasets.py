from typing import List

import pytest

from deepfake_detection.data.datasets.fileimagedataset import FileImageDataset
from deepfake_detection.data.datasets.filevideodataset import FileVideoDataset
from deepfake_detection.data.datasets.fileimagesequencedataset import FileImageSequenceDataset
from deepfake_detection.data.instance import Instance
from deepfake_detection.models.prediction import Prediction
from tests.deepfake_detection.paths import RESOURCES_DIR


@pytest.fixture
def instances() -> List[Instance]:
    return [Instance("", labels) for labels in (
        {"A"},
        {"A"},
        {"A", "B", "C"},
        {"A", "C"},
        {"B", "C"},
        {"B", "C"},
        {"B", "C"},
        {"C"}
    )]


@pytest.fixture
def predictions() -> List[List[Prediction]]:
    return [
        [Prediction(classification=classification)]
        for classification in [
            {"A": 1.0, "B": 0.0, "C": 0.1},  # Ground truth: "A"
            {"A": 0.5, "B": 0.4, "C": 0.3},  # Ground truth: "A"
            {"A": 1.0, "B": 0.9, "C": 0.0},  # Ground truth: "ABC"
            {"A": 1.0, "B": 0.6, "C": 0.5},  # Ground truth: "AC"
            {"A": 0.0, "B": 0.7, "C": 0.5},  # Ground truth: "BC"
            {"A": 0.0, "B": 1.0, "C": 0.9},  # Ground truth: "BC"
            {"A": 0.1, "B": 0.7, "C": 0.4},  # Ground truth: "BC"
            {"A": 0.7, "B": 0.5, "C": 0.6},  # Ground truth: "C"
        ]
    ]


@pytest.fixture
def image_dataset_path():
    return RESOURCES_DIR / "data" / "test_image_dataset"


@pytest.fixture
def image_sequence_dataset_path():
    return RESOURCES_DIR / "data" / "test_image_sequence_dataset"


@pytest.fixture
def video_dataset_path():
    return RESOURCES_DIR / "data" / "test_video_dataset"


def test_load_file_image_dataset(image_dataset_path):
    dataset = FileImageDataset(name='test', path=image_dataset_path)
    instances = list(dataset)

    assert len(dataset) == 3
    assert len(dataset) == len(instances)


def test_load_file_image_sequence_dataset(image_sequence_dataset_path):
    dataset = FileImageSequenceDataset(name='test', path=image_sequence_dataset_path)
    instances = list(dataset)

    assert len(dataset) == 6
    assert len(dataset) == len(instances)


def test_load_video_dataset(video_dataset_path):
    dataset = FileVideoDataset(name='test', path=video_dataset_path)
    instances = list(dataset)

    assert len(dataset) == 3
    assert len(dataset) == len(instances)


def test_hash_dataset_same_name_equal(image_dataset_path):
    dataset = FileImageDataset(name='test', path=image_dataset_path)
    dataset2 = FileImageDataset(name='test', path=image_dataset_path)
    assert hash(dataset) == hash(dataset2)


def test_hash_dataset_different_name_different(image_dataset_path):
    dataset = FileImageDataset(name='test', path=image_dataset_path)
    dataset2 = FileImageDataset(name='test2', path=image_dataset_path)
    assert hash(dataset) != hash(dataset2)