from typing import List

import pytest

from deepfake_detection.data.datasets.instance import Instance
from deepfake_detection.evaluation.metrics import accuracy
from deepfake_detection.models.prediction import Prediction


@pytest.fixture
def instances() -> List[Instance]:
    return [Instance("", class_label=c_label.upper(), authenticity_label=a_label.upper())
            for (a_label, c_label) in zip("rrrrffff", "aaaabbcc")]


@pytest.fixture
def predictions() -> List[Prediction]:
    return [
        Prediction(classification=classification)
        for classification in [
            {"A": 0.8, "B": 0.1, "C": 0.1},  # True = "A" hit
            {"A": 0.4, "B": 0.5, "C": 0.1},  # True = "A"
            {"A": 0.7, "B": 0.0, "C": 0.3},  # True = "A" hit
            {"A": 0.8, "B": 0.1, "C": 0.1},  # True = "A" hit
            {"A": 0.1, "B": 0.7, "C": 0.2},  # True = "B" hit
            {"A": 0.5, "B": 0.2, "C": 0.3},  # True = "B"
            {"A": 0.6, "B": 0.4, "C": 0.0},  # True = "C"
            {"A": 0.0, "B": 0.3, "C": 0.7},  # True = "C" hit
        ]
    ]


def test_accuracy_authenticity_label(instances, predictions):
    actual = accuracy(instances, predictions, positive_label='F')
    assert pytest.approx(actual) == 4 / 8


def test_accuracy_class_label(instances, predictions):
    actual = accuracy(instances, predictions, label_type='class_label', positive_label='A')
    assert pytest.approx(actual) == 5 / 8

def test_accuracy_invalid_label(instances, predictions):
    with pytest.raises(ValueError):
        actual = accuracy(instances, predictions, label_type='other', positive_label='A')