import pytest

from deepfake_detection.evaluation.metrics import accuracy, roc_auc
from tests.deepfake_detection.evaluation.config import source_predictions, authenticity_predictions, instances


def test_accuracy_source_label_specified(instances, source_predictions):
    actual = accuracy(instances, source_predictions, label='A', label_type='source_label')
    assert pytest.approx(actual) == 5 / 8


def test_accuracy_authenticity_label_specified(instances, authenticity_predictions):
    actual = accuracy(instances, authenticity_predictions, label='fake', label_type='authenticity_label')
    assert pytest.approx(actual) == 5 / 8


def test_accuracy_authenticity_source_label_specified(instances, authenticity_predictions):
    actual = accuracy(instances, authenticity_predictions, label='A', label_type='authenticity_label')
    assert pytest.approx(actual) == 3 / 4


def test_accuracy_source_label_not_specified(instances, source_predictions):
    actual = accuracy(instances, source_predictions, label_type='source_label')
    assert pytest.approx(actual) == 6 / 8


def test_accuracy_authenticity_label_not_specified(instances, authenticity_predictions):
    actual = accuracy(instances, authenticity_predictions, label_type='authenticity_label')
    assert pytest.approx(actual) == 5 / 8


def test_accuracy_different_length(instances, source_predictions):
    with pytest.raises(ValueError):
        accuracy(instances, source_predictions[:-1], label_type='source_label')


def test_roc_auc_source_label_specified(instances, source_predictions):
    actual = roc_auc(instances, source_predictions, label='A', label_type='source_label')
    assert pytest.approx(actual) == 0.875


def test_roc_auc_authenticity_label_specified(instances, authenticity_predictions):
    actual = roc_auc(instances, authenticity_predictions, label='fake', label_type='authenticity_label')
    assert pytest.approx(actual) == 0.8125


def test_roc_auc_label_not_in_label_type_error(instances, authenticity_predictions):
    with pytest.raises(ValueError):
        roc_auc(instances, authenticity_predictions, label='A', label_type='authenticity_label')


def test_roc_auc_source_label_not_specified(instances, source_predictions):
    actual = roc_auc(instances, source_predictions, label_type='source_label')
    assert pytest.approx(actual) == 0.708333


def test_roc_auc_different_length(instances, source_predictions):
    with pytest.raises(ValueError):
        roc_auc(instances, source_predictions[:-1], label_type='source_label')