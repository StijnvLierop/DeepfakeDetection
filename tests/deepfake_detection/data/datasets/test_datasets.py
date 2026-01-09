import os
import pytest

from deepfake_detection.data.datasets import (ListDataset, FileImageSequenceDataset, FileImageDataset,
                                              FileVideoDataset, FilteredDataset)
from tests.deepfake_detection.paths import RESOURCES_DIR


@pytest.fixture
def image_split_file_path():
    return os.path.join(RESOURCES_DIR, "data", "test_image_dataset", "split_file.txt")


@pytest.fixture
def image_sequence_split_file_path():
    return os.path.join(
        RESOURCES_DIR, "data", "test_image_sequence_dataset", "split_file.txt"
    )


@pytest.fixture
def video_split_file_path():
    return os.path.join(RESOURCES_DIR, "data", "test_video_dataset", "split_file.txt")


def test_load_file_image_dataset(image_dataset_path):
    dataset = FileImageDataset(name="test", path=image_dataset_path)
    instances = list(dataset)

    assert len(dataset) == 3
    assert len(dataset) == len(instances)


def test_load_file_image_dataset_split(image_dataset_path, image_split_file_path):
    dataset = FileImageDataset(
        name="test", path=image_dataset_path, split_file=image_split_file_path
    )
    instances = list(dataset)

    assert len(dataset) == 1
    assert len(dataset) == len(instances)


def test_load_file_image_sequence_dataset(image_sequence_dataset_path):
    dataset = FileImageSequenceDataset(name="test", path=image_sequence_dataset_path)
    instances = list(dataset)

    assert len(dataset) == 6
    assert len(dataset) == len(instances)


def test_load_file_image_sequence_dataset_split(
    image_sequence_dataset_path, image_sequence_split_file_path
):
    dataset = FileImageSequenceDataset(
        name="test",
        path=image_sequence_dataset_path,
        split_file=image_sequence_split_file_path,
    )
    instances = list(dataset)

    assert len(dataset) == 1
    assert len(dataset) == len(instances)


def test_load_video_dataset(video_dataset_path):
    dataset = FileVideoDataset(name="test", path=video_dataset_path)
    instances = list(dataset)

    assert len(dataset) == 3
    assert len(dataset) == len(instances)


def test_load_file_image_video_dataset_split(video_dataset_path, video_split_file_path):
    dataset = FileVideoDataset(
        name="test", path=video_dataset_path, split_file=video_split_file_path
    )
    instances = list(dataset)

    assert len(dataset) == 1
    assert len(dataset) == len(instances)


def test_dataset_same_instances_equal(image_dataset_path):
    dataset = FileImageDataset(name="test", path=image_dataset_path)
    dataset2 = FileImageDataset(name="test2", path=image_dataset_path)
    assert dataset == dataset2
    dataset3 = ListDataset(name="test3", instances=list(dataset))
    assert dataset == dataset3


def test_dataset_different_instances_different(image_dataset_path):
    dataset = FileImageDataset(name="test", path=image_dataset_path)
    dataset2 = ListDataset(name="test3", instances=list(dataset)[:-1])
    assert dataset != dataset2


def test_filtered_dataset_iteration(dummy_dataset):
    def indices(dataset):
        return [2, 4]
    filtered_dataset = FilteredDataset(dummy_dataset, indices)
    result = list(filtered_dataset)
    answer = [list(dummy_dataset)[2], list(dummy_dataset)[4]]
    assert result == answer


def test_filtered_dataset_empty_indices(dummy_dataset):
    def indices(dataset):
        return []
    filtered_dataset = FilteredDataset(dummy_dataset, indices)
    result = list(filtered_dataset)
    assert result == []


def test_filtered_dataset_length(dummy_dataset):
    def indices(dataset):
        return [1, 5]
    filtered_dataset = FilteredDataset(dummy_dataset, indices)
    assert len(filtered_dataset) == len(list(filtered_dataset))
    assert 2 == len(filtered_dataset)


def test_iter_yields_batches_correctly(dummy_dataset):
    batch_size = 3
    batches = list(dummy_dataset.iter(batch_size))

    # Check the number of batches
    expected_num_batches = (len(dummy_dataset) + batch_size - 1) // batch_size
    assert len(batches) == expected_num_batches

    # Check individual batches
    for i, batch in enumerate(batches):
        if i == len(batches) - 1:
            # The last batch might be smaller than the specified batch_size
            assert len(batch) == len(dummy_dataset) % batch_size or batch_size
        else:
            assert len(batch) == batch_size

    # Ensure all elements from the dataset are yielded
    flattened_batches = [item for batch in batches for item in batch]
    assert dummy_dataset == flattened_batches


def test_iter_empty_dataset():
    empty_dataset = ListDataset([])
    batches = empty_dataset.iter(batch_size=5)
    assert list(batches) == []


def test_iter_with_large_batch_size(dummy_dataset):
    batch_size = 40
    batches = list(dummy_dataset.iter(batch_size))

    # The entire dataset should be yielded as a single batch
    assert len(batches) == 1
    assert len(batches[0]) == len(dummy_dataset)


def test_get_item(dummy_dataset):
    instance = dummy_dataset[0]
    assert instance.annotation.authenticity_label == "real"
