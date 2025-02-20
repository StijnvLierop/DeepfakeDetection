import pytest

from deepfake_detection.evaluation.metrics import accuracy, roc_auc
from tests.deepfake_detection.evaluation.config import predictions, instances


def test_accuracy_label_specified(instances, predictions):
    actual = accuracy(instances, predictions, label='A')
    assert pytest.approx(actual) == 5 / 8


def test_accuracy_label_not_specified(instances, predictions):
    actual = accuracy(instances, predictions)
    assert pytest.approx(actual) == 6 / 8


def test_accuracy_different_length(instances, predictions):
    with pytest.raises(ValueError):
        accuracy(instances, predictions[:-1])


def test_accuracy_label_mapping(instances, predictions):
    actual = accuracy(instances,
                      predictions,
                      label_mapping={"A": "F", "B": "R", "C": "F"}
                      )
    assert pytest.approx(actual) == 0.75


def test_accuracy_label_mapping_label_specified(instances, predictions):
    actual = accuracy(instances,
                      predictions,
                      label='F',
                      label_mapping={"A": "R", "B": "R", "C": "F"}
                      )
    assert pytest.approx(actual) == 0.875


def test_accuracy_label_mapping_fake_predictions(instances, predictions):
    predictions[0].classification = {"A": 0.5, "B": 0.2, "C": 0.2, "R": 0.7}
    actual = accuracy(instances,
                      predictions,
                      label_mapping={"A": "F", "B": "R", "C": "F"}
                      )
    assert pytest.approx(actual) == 0.625


def test_roc_auc_label_specified(instances, predictions):
    actual = roc_auc(instances, predictions, label='A')
    assert pytest.approx(actual) == 0.875


def test_roc_auc_label_not_specified(instances, predictions):
    actual = roc_auc(instances, predictions)
    assert pytest.approx(actual) == 0.708333


def test_roc_auc_different_length(instances, predictions):
    with pytest.raises(ValueError):
        roc_auc(instances, predictions[:-1])


def test_roc_auc_label_mapping(instances, predictions):
    actual = roc_auc(instances,
                     predictions,
                     label_mapping={"A": "F", "B": "R", "C": "F"}
                     )
    assert pytest.approx(actual) == 0.833333


def test_roc_auc_label_mapping_label_specified(instances, predictions):
    actual = roc_auc(instances,
                     predictions,
                     label='F',
                     label_mapping={"A": "R", "B": "R", "C": "F"}
                     )
    assert pytest.approx(actual) == 0.5


def test_roc_auc_label_mapping_fake_predictions(instances, predictions):
    predictions[0].classification = {"A": 0.5, "B": 0.2, "C": 0.2, "R": 0.7}
    actual = roc_auc(instances,
                     predictions,
                     label_mapping={"A": "F", "B": "R", "C": "F"}
                     )
    assert pytest.approx(actual) == 0.75