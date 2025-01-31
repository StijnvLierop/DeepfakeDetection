import pytest

from deepfake_detection.data.datasets.FileImageDataset import FileImageDataset
from deepfake_detection.data.datasets.FileImageSequenceDataset import FileImageSequenceDataset
from tests.deepfake_detection.paths import RESOURCES_DIR


@pytest.fixture
def image_dataset_path():
    return RESOURCES_DIR / "data" / "test_image_dataset"


@pytest.fixture
def image_sequence_dataset_path():
    return RESOURCES_DIR / "data" / "test_image_sequence_dataset"


def test_load_file_image_dataset(image_dataset_path):
    dataset = FileImageDataset(name='test', path=image_dataset_path)
    instances = list(dataset)

    assert len(dataset) == 3
    assert len(dataset) == len(instances)
    assert instances[0].label == 'real'
    assert instances[1].label == 'model2'
    assert instances[2].label == 'model1'


def test_load_file_image_sequence_dataset(image_sequence_dataset_path):
    dataset = FileImageSequenceDataset(name='test', path=image_sequence_dataset_path)
    instances = list(dataset)

    assert len(dataset) == 6
    assert len(dataset) == len(instances)
    assert instances[0].label == 'real'
    assert len(instances[0]) == 4
    assert instances[2].label == 'model2'
    assert len(instances[1]) == 4
    assert instances[4].label == 'model1'
    assert len(instances[2]) == 4
