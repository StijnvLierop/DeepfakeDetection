import numpy as np
from deepfake_detection.evaluation.utils import get_labels, to_arrays, map_fields
from tests.deepfake_detection.evaluation.config import instances, predictions


def test_get_labels(instances, predictions):
    expected_labels = ["A", "B", "C"]
    assert get_labels(instances, predictions) == expected_labels

def test_to_arrays(instances, predictions):
    y_true, y_pred = to_arrays(instances, predictions, "B")
    np.testing.assert_array_equal(y_true, np.array([0, 0, 0, 0, 1, 1, 0, 0]))
    np.testing.assert_array_equal(y_pred, np.array([0.1, 0.5, 0.0, 0.1, 0.7, 0.2, 0.4, 0.3]))

def test_to_arrays_binary(instances, predictions):
    y_true, y_pred = to_arrays(instances, predictions, "B", binary=True)
    np.testing.assert_array_equal(y_true, np.array([0, 0, 0, 0, 1, 1, 0, 0]))
    np.testing.assert_array_equal(y_pred, np.array([0, 1, 0, 0, 1, 0, 0, 0]))

def test_map_fields():
    init_dict = {"a": 1.0, "b": 2.0, "c": 0.5}
    map_dict = {"a": "x", "b": "y", "c": "x"}
    expected_output = {"x": 1.0, "y": 2.0}  # "x" should take max of 1_1fake.1_0fake and 1_0fake.5
    assert map_fields(init_dict, map_dict) == expected_output
