import os
from typing import List

import pytest

from deepfake_detection.data.datasets import ListDataset
from deepfake_detection.data.datasets.fileimagedataset import FileImageDataset
from deepfake_detection.data.datasets.filevideodataset import FileVideoDataset
from deepfake_detection.data.datasets.fileimagesequencedataset import FileImageSequenceDataset
from deepfake_detection.data.instance import Instance
from deepfake_detection.models.prediction import Prediction
from tests.deepfake_detection.fixtures import image_dataset_path, image_sequence_dataset_path, video_dataset_path
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
def image_split_file_path():
    return os.path.join(RESOURCES_DIR, "data", "test_image_dataset", "split_file.txt")


@pytest.fixture
def image_sequence_split_file_path():
    return os.path.join(RESOURCES_DIR, "data", "test_image_sequence_dataset", "split_file.txt")


@pytest.fixture
def video_split_file_path():
    return os.path.join(RESOURCES_DIR, "data", "test_video_dataset", "split_file.txt")


def test_load_file_image_dataset(image_dataset_path):
    dataset = FileImageDataset(name='test', path=image_dataset_path)
    instances = list(dataset)

    assert len(dataset) == 3
    assert len(dataset) == len(instances)


def test_load_file_image_dataset_split(image_dataset_path, image_split_file_path):
    dataset = FileImageDataset(name='test', path=image_dataset_path, split_file=image_split_file_path)
    instances = list(dataset)

    assert len(dataset) == 1
    assert len(dataset) == len(instances)


def test_load_file_image_sequence_dataset(image_sequence_dataset_path):
    dataset = FileImageSequenceDataset(name='test', path=image_sequence_dataset_path)
    instances = list(dataset)

    assert len(dataset) == 6
    assert len(dataset) == len(instances)


def test_load_file_image_sequence_dataset_split(image_sequence_dataset_path, image_sequence_split_file_path):
    dataset = FileImageSequenceDataset(name='test',
                                       path=image_sequence_dataset_path,
                                       split_file=image_sequence_split_file_path)
    instances = list(dataset)

    assert len(dataset) == 1
    assert len(dataset) == len(instances)


def test_load_video_dataset(video_dataset_path):
    dataset = FileVideoDataset(name='test', path=video_dataset_path)
    instances = list(dataset)

    assert len(dataset) == 3
    assert len(dataset) == len(instances)


def test_load_file_image_video_dataset_split(video_dataset_path, video_split_file_path):
    dataset = FileVideoDataset(name='test', path=video_dataset_path, split_file=video_split_file_path)
    instances = list(dataset)

    assert len(dataset) == 1
    assert len(dataset) == len(instances)


def test_dataset_same_instances_equal(image_dataset_path):
    dataset = FileImageDataset(name='test', path=image_dataset_path)
    dataset2 = FileImageDataset(name='test2', path=image_dataset_path)
    assert dataset == dataset2
    dataset3 = ListDataset(name='test3', instances=list(dataset))
    assert dataset == dataset3


def test_dataset_different_instances_different(image_dataset_path):
    dataset = FileImageDataset(name='test', path=image_dataset_path)
    dataset2 = ListDataset(name='test3', instances=list(dataset)[:-1])
    assert dataset != dataset2