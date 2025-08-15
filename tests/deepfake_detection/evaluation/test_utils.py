import numpy as np
import pytest

from deepfake_detection.evaluation.utils import get_labels, to_arrays, map_fields, \
    find_label_type_corresponding_with_label
from tests.deepfake_detection.evaluation.config import instances, source_predictions


def test_get_labels(instances, source_predictions):
    expected_labels = ["A", "B", "C"]
    assert get_labels(instances, source_predictions, label_type='source_label') == expected_labels


def test_get_labels_warning_when_no_overlap_in_labels(instances, source_predictions):
    with pytest.warns():
        get_labels(instances, source_predictions)


def test_to_arrays(instances, source_predictions):
    y_true, y_pred = to_arrays(instances, source_predictions, "B", label_type='source_label')
    np.testing.assert_array_equal(y_true, np.array([0, 0, 0, 0, 1, 1, 0, 0]))
    np.testing.assert_array_equal(y_pred, np.array([0.1, 0.5, 0.0, 0.1, 0.7, 0.2, 0.4, 0.3]))


def test_to_arrays_binary(instances, source_predictions):
    y_true, y_pred = to_arrays(instances, source_predictions, "B", label_type='source_label', binary=True)
    np.testing.assert_array_equal(y_true, np.array([0, 0, 0, 0, 1, 1, 0, 0]))
    np.testing.assert_array_equal(y_pred, np.array([0, 1, 0, 0, 1, 0, 0, 0]))


def test_to_arrays_warning_when_no_overlap_in_labels(instances, source_predictions):
    with pytest.warns():
        to_arrays(instances, source_predictions, "B", label_type='authenticity_label', binary=True)


def test_map_fields():
    init_dict = {"a": 1.0, "b": 2.0, "c": 0.5}
    map_dict = {"a": "x", "b": "y", "c": "x"}
    expected_output = {"x": 1.0, "y": 2.0}  # "x" should take max of 1_1fake.1_0fake and 1_0fake.5
    assert map_fields(init_dict, map_dict) == expected_output


def test_find_label_type_corresponding_with_label(instances):
    assert find_label_type_corresponding_with_label(instances, "real") == 'authenticity_label'
    assert find_label_type_corresponding_with_label(instances, "A") == 'source_label'
    assert find_label_type_corresponding_with_label(instances, 0) == 'binary_label'


def test_find_label_type_corresponding_with_label_unknown_label(instances):
    with pytest.raises(ValueError):
        find_label_type_corresponding_with_label(instances, "unknown label")