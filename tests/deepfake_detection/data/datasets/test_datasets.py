import os

import pytest

from deepfake_detection.data.datasets import ListDataset
from deepfake_detection.data.datasets.fileimagedataset import FileImageDataset
from deepfake_detection.data.datasets.filevideodataset import FileVideoDataset
from deepfake_detection.data.datasets.fileimagesequencedataset import FileImageSequenceDataset
from deepfake_detection.data.datasets.filtered_dataset import FilteredDataset
from tests.deepfake_detection.fixtures import (image_dataset_path, image_sequence_dataset_path,
                                               video_dataset_path, dummy_dataset)
from tests.deepfake_detection.paths import RESOURCES_DIR


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


def test_filtered_dataset_iteration(dummy_dataset):
    indices = [2, 4]
    filtered_dataset = FilteredDataset(dummy_dataset, indices)
    result = list(filtered_dataset)
    answer = [list(dummy_dataset)[2], list(dummy_dataset)[4]]
    assert result == answer


def test_filtered_dataset_empty_indices(dummy_dataset):
    indices = []
    filtered_dataset = FilteredDataset(dummy_dataset, indices)
    result = list(filtered_dataset)
    assert result == []


def test_filtered_dataset_length(dummy_dataset):
    indices = [1, 5]
    filtered_dataset = FilteredDataset(dummy_dataset, indices)
    assert len(filtered_dataset) == len(list(filtered_dataset))
    assert len(indices) == len(filtered_dataset)