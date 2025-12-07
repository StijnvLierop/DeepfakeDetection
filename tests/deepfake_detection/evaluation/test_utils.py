import numpy as np
import pytest

from deepfake_detection.evaluation.utils import get_labels, to_arrays, map_fields, \
    find_label_type_corresponding_with_label, get_label_mapping, transform_prediction
from deepfake_detection.models import Prediction
from tests.deepfake_detection.evaluation.config import instances, source_predictions


@pytest.fixture
def dummy_prediction():
    classification = {
        "class_1": 0.8,
        "class_2": 0.2
    }
    return Prediction(classification=classification)


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


def test_label_mapping_source_to_source(instances):
    mapping = get_label_mapping(instances, source_label='source_label', target_label='source_label')
    assert mapping == {'A': 'A', 'B': 'B', 'C': 'C'}


def test_label_mapping_source_to_authenticity(instances):
    mapping = get_label_mapping(instances, source_label='source_label', target_label='authenticity_label')
    assert mapping == {'A': 'real', 'B': 'fake', 'C': 'fake'}


def test_label_mapping_source_to_binary(instances):
    mapping = get_label_mapping(instances, source_label='source_label', target_label='binary_label')
    assert mapping == {'A': 0, 'B': 1, 'C': 1}


def test_label_mapping_invalid_labels(instances):
    with pytest.raises(ValueError):
        get_label_mapping(instances, source_label='test_label', target_label='source_label')
    with pytest.raises(ValueError):
        get_label_mapping(instances, source_label='source_label', target_label='test_label')


def test_transform_prediction_updates_classification(dummy_prediction):
    label_mapping = {
        "class_1": "mapped_class_1",
        "class_2": "mapped_class_1"
    }
    expected_classification = {
        "mapped_class_1": 1.0
    }
    transformed = transform_prediction(dummy_prediction, label_mapping)
    assert expected_classification == transformed.classification


def test_transform_prediction_keeps_unmapped_labels(dummy_prediction):
    label_mapping = {
        "class_2": "mapped_class_2"
    }
    expected_classification = {
        "class_1": 0.8,
        "mapped_class_2": 0.2
    }
    transformed = transform_prediction(dummy_prediction, label_mapping)
    assert expected_classification == transformed.classification


def test_transform_prediction_no_labels_changed(dummy_prediction):
    label_mapping = {}
    expected_classification = dummy_prediction.classification
    transformed = transform_prediction(dummy_prediction, label_mapping)
    assert expected_classification == transformed.classification


def test_transform_prediction_with_empty_classification():
    empty_prediction = Prediction(classification={})
    label_mapping = {
        "class_1": "mapped_class_1"
    }
    expected_classification = {}
    transformed = transform_prediction(empty_prediction, label_mapping)
    assert expected_classification == transformed.classification