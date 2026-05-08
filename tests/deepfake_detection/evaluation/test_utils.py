import numpy as np
import pytest

from deepfake_detection.evaluation.utils import (
    get_labels,
    to_arrays,
    map_fields,
    transform_prediction,
)
from deepfake_detection.models import Prediction


@pytest.fixture
def dummy_prediction():
    classification = {"class_1": 0.8, "class_2": 0.2}
    return Prediction(classification=classification)


def test_get_labels(instances, source_predictions):
    expected_labels = ["A", "B", "C"]
    assert (
        get_labels(instances, source_predictions, label_type="source")
        == expected_labels
    )


def test_get_labels_warning_when_no_overlap_in_labels(instances, source_predictions):
    with pytest.warns():
        get_labels(instances, source_predictions, label_type="authenticity")


def test_to_arrays(instances, source_predictions):
    y_true, y_pred = to_arrays(instances, source_predictions, "B", label_type="source")
    np.testing.assert_array_equal(y_true, np.array([0, 0, 0, 0, 1, 1, 0, 0]))
    np.testing.assert_array_equal(
        y_pred, np.array([0.1, 0.5, 0.0, 0.1, 0.7, 0.2, 0.4, 0.3])
    )


def test_to_arrays_binary(instances, source_predictions):
    y_true, y_pred = to_arrays(
        instances, source_predictions, "B", label_type="source", binary=True
    )
    np.testing.assert_array_equal(y_true, np.array([0, 0, 0, 0, 1, 1, 0, 0]))
    np.testing.assert_array_equal(y_pred, np.array([0, 1, 0, 0, 1, 0, 0, 0]))


def test_to_arrays_warning_when_no_overlap_in_labels(instances, source_predictions):
    with pytest.warns():
        to_arrays(
            instances,
            source_predictions,
            "B",
            label_type="authenticity",
            binary=True,
        )


def test_map_fields():
    init_dict = {"a": 1.0, "b": 2.0, "c": 0.5}
    map_dict = {"a": "x", "b": "y", "c": "x"}
    expected_output = {
        "x": 1.0,
        "y": 2.0,
    }  # "x" should take max of 1_1fake.1_0fake and 1_0fake.5
    assert map_fields(init_dict, map_dict) == expected_output


def test_transform_prediction_updates_classification(dummy_prediction):
    label_mapping = {"class_1": "mapped_class_1", "class_2": "mapped_class_1"}
    expected_classification = {"mapped_class_1": 1.0}
    transformed = transform_prediction(dummy_prediction, label_mapping)
    assert expected_classification == transformed.classification


def test_transform_prediction_keeps_unmapped_labels(dummy_prediction):
    label_mapping = {"class_2": "mapped_class_2"}
    expected_classification = {"class_1": 0.8, "mapped_class_2": 0.2}
    transformed = transform_prediction(dummy_prediction, label_mapping)
    assert expected_classification == transformed.classification


def test_transform_prediction_no_labels_changed(dummy_prediction):
    label_mapping = {}
    expected_classification = dummy_prediction.classification
    transformed = transform_prediction(dummy_prediction, label_mapping)
    assert expected_classification == transformed.classification


def test_transform_prediction_with_empty_classification():
    empty_prediction = Prediction(classification={})
    label_mapping = {"class_1": "mapped_class_1"}
    expected_classification = {}
    transformed = transform_prediction(empty_prediction, label_mapping)
    assert expected_classification == transformed.classification
