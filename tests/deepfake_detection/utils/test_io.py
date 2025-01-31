import tempfile

import numpy as np
import pytest

from deepfake_detection.data.datasets.FileImageDataset import FileImageDataset
from deepfake_detection.utils.io import write_predictions_to_file, read_predictions_from_file
from tests.deepfake_detection.paths import RESOURCES_DIR


@pytest.fixture
def image_dataset():
    return FileImageDataset(RESOURCES_DIR / "data" / "test_image_dataset", name="test_image_dataset")


@pytest.fixture
def predictions(image_dataset):
    return list(np.arange(len(image_dataset), dtype=float) / 10)


def test_write_read_predictions_to_file(image_dataset, predictions):
    temp_dir = tempfile.gettempdir()
    outfile_path = write_predictions_to_file(temp_dir, predictions, image_dataset, model_name="test_model")
    read_predictions = read_predictions_from_file(outfile_path)['prediction'].to_list()
    assert read_predictions == predictions


def test_write_predictions_to_file_wrong_length(image_dataset, predictions):
    temp_dir = tempfile.gettempdir()
    pytest.raises(ValueError,
                  write_predictions_to_file,
                  temp_dir,
                  predictions[:-2],
                  image_dataset,
                  model_name="test_model")


def get_predictions_filename():
    name = get_predictions_filename('testdataset', 'testmodel')
    assert name == f'predictions_testdataset_testmodel.csv'