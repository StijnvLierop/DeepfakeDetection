from pathlib import Path

import numpy as np
import pytest
from deepfake_detection.data.annotation import Annotation
from deepfake_detection.data.instance import Instance
from deepfake_detection.evaluation.confusion_matrix import confusion_matrix
from deepfake_detection.models.prediction import Prediction


class MockInstance(Instance):
    def save(self, path: Path) -> Path:
        pass

    def __init__(self, annotation):
        super().__init__(annotation)

    def __hash__(self):
        return hash(self.annotation["authenticity"])


@pytest.fixture
def test_instances():
    return [
        MockInstance(Annotation({"authenticity": "fake", "source": "genA"})),
        MockInstance(Annotation({"authenticity": "real", "source": "cam1"})),
        MockInstance(Annotation({"authenticity": "fake", "source": "genB"})),
    ]


def test_confusion_matrix(test_instances):
    predictions = [
        Prediction(classification={"fake": 0.9, "real": 0.1}),
        Prediction(classification={"fake": 0.2, "real": 0.8}),
        Prediction(classification={"fake": 0.3, "real": 0.7}),
    ]
    result = confusion_matrix(test_instances, predictions, "authenticity")
    expected = [[1, 1], [0, 1]]
    assert result.shape == (2, 2)
    assert np.array_equal(result, expected)


def test_confusion_matrix_mismatched_lengths(test_instances):
    predictions = [Prediction(classification={"fake": 0.9, "real": 0.1})]
    with pytest.raises(
        ValueError, match="Predictions and instances must have the same length."
    ):
        confusion_matrix(test_instances, predictions, "authenticity")


def test_confusion_matrix_source_label(test_instances):
    predictions = [
        Prediction(classification={"genA": 0.99, "genB": 0.01, "cam1": 0.0}),
        Prediction(classification={"genA": 0.7, "genB": 0.21, "cam1": 0.09}),
        Prediction(classification={"genA": 0.5, "genB": 0.3, "cam1": 0.2}),
    ]
    result = confusion_matrix(test_instances, predictions, "source")
    expected = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert result.shape == (3, 3)
    assert np.array_equal(result, expected)


def test_confusion_matrix_no_common_labels(test_instances):
    predictions = [
        Prediction(classification={"genA": 0.99, "genB": 0.01, "cam1": 0.0}),
        Prediction(classification={"genA": 0.7, "genB": 0.21, "cam1": 0.09}),
        Prediction(classification={"genA": 0.5, "genB": 0.3, "cam1": 0.2}),
    ]
    with pytest.warns(UserWarning, match="No common labels between y_pred and y_true."):
        confusion_matrix(test_instances, predictions, "authenticity")
