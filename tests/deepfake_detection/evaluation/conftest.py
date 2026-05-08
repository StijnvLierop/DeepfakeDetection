from typing import List

import numpy as np
import pytest
from PIL import Image

from deepfake_detection.data.annotation import Annotation
from deepfake_detection.data.instance import Instance, ImageInstance
from deepfake_detection.models.prediction import Prediction


@pytest.fixture
def instances() -> List[Instance]:
    return [
        ImageInstance(
            data=Image.fromarray(np.zeros(10)),
            annotation=Annotation({"authenticity": a_label, "source": c_label}),
        )
        for (a_label, c_label) in zip(
            ["real", "real", "real", "real", "fake", "fake", "fake", "fake"], "AAAABBCC"
        )  # A is real, B and C are fake
    ]


@pytest.fixture
def source_predictions() -> List[Prediction]:
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


@pytest.fixture
def authenticity_predictions() -> List[Prediction]:
    return [
        Prediction(classification=classification)
        for classification in [
            {"real": 0.8, "fake": 0.2},  # True = "r" hit
            {"real": 0.4, "fake": 0.6},  # True = "r"
            {"real": 1.0, "fake": 0.0},  # True = "r" hit
            {"real": 0.8, "fake": 0.2},  # True = "r" hit
            {"real": 0.3, "fake": 0.7},  # True = "f" hit
            {"real": 0.8, "fake": 0.2},  # True = "f"
            {"real": 0.2, "fake": 0.8},  # True = "f" hit
            {"real": 0.7, "fake": 0.3},  # True = "f"
        ]
    ]
