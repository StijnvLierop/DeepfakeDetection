import tempfile
from pathlib import Path

import numpy as np
import pytest

from deepfake_detection.data.datasets.fileimagedataset import FileImageDataset
from deepfake_detection.models.prediction import Prediction
from deepfake_detection.utils.io import write_predictions_to_file, read_predictions_from_file, jsonify, \
    encode_prediction, decode_prediction
from tests.deepfake_detection.paths import RESOURCES_DIR


@pytest.fixture
def image_dataset():
    return FileImageDataset(RESOURCES_DIR / "data" / "test_image_dataset", name="test_image_dataset")


@pytest.fixture
def predictions(image_dataset):
    return [Prediction(classification={'score':0.1}), Prediction(embedding=[0.1,0.2]), Prediction(text='hoi')]


@pytest.fixture()
def prediction() -> Prediction:
    return Prediction(
        classification={"A": 0.8, "B": 0.2},
        embedding=[1, 2, 3, 4],
        text="Test",
        image=np.array([1, 1]),
    )

@pytest.fixture
def unserializable():
    res = iter([1, 2])
    # Make sure our implementation of `jsonify` has not changed and this still
    # produces an error.
    with pytest.raises(TypeError):
        jsonify(res)
    return res


def test_write_read_predictions_to_file(image_dataset, predictions):
    temp_dir = tempfile.gettempdir()
    filepath = f"{temp_dir}/predictions.json"
    write_predictions_to_file(predictions, Path(filepath))
    read_predictions = read_predictions_from_file(filepath)
    assert read_predictions == predictions


def test_write_read_predictions_to_file_parent_dir_not_exists(image_dataset, predictions):
    temp_dir = tempfile.gettempdir()
    filepath = f"{temp_dir}/parent/predictions.json"
    write_predictions_to_file(predictions, Path(filepath))
    read_predictions = read_predictions_from_file(filepath)
    assert read_predictions == predictions


def test_write_read_predictions_to_file_parent_parent_dir_not_exists(image_dataset, predictions):
    temp_dir = tempfile.gettempdir()
    filepath = f"{temp_dir}/parent/predictions.json"
    write_predictions_to_file(predictions, Path(filepath))
    read_predictions = read_predictions_from_file(filepath)
    assert read_predictions == predictions


def get_predictions_filename():
    name = get_predictions_filename('testdataset', 'testmodel')
    assert name == f'predictions_testdataset_testmodel.json'


def test_encode_decode_prediction(prediction):
    encoded = encode_prediction(prediction)
    decoded = decode_prediction(encoded)
    assert prediction == decoded


def test_encode_decode_prediction_with_meta():
    meta = {"foo": "bar"}
    prediction = Prediction(classification={"A": 0.8, "B": 0.2}, meta=meta)
    encoded = encode_prediction(prediction)
    assert encoded["meta"]["foo"] == "bar"
    decoded = decode_prediction(encoded)
    assert prediction == decoded


def test_encode_decode_prediction_skips_unserializable_meta(unserializable):
    meta = {
        "unserializable": unserializable,  # Should not be serialized
        "foo": "bar",
    }
    prediction = Prediction(classification={"A": 0.8, "B": 0.2}, meta=meta)
    with pytest.warns(UserWarning, match="unserializable"):
        encoded = encode_prediction(prediction)
        assert encoded["meta"] == {"foo": "bar"}


def test_json_encode_decode_prediction_with_unserializable_meta(
        unserializable
):
    meta = {
        "unserializable": unserializable,
        "foo": "bar",
    }
    prediction = Prediction(classification={"A": 0.8, "B": 0.2}, meta=meta)
    with pytest.warns(UserWarning, match="unserializable"):
        encoded = encode_prediction(prediction)
        decoded = decode_prediction(encoded)
        assert prediction == decoded