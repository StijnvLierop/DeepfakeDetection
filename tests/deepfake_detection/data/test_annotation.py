import pytest

from deepfake_detection.data.annotation import Annotation


def test_binary_label():
    annot = Annotation(authenticity_label="fake")
    assert annot.binary_label == 1
    annot = Annotation(authenticity_label="real")
    assert annot.binary_label == 0
    annot = Annotation(authenticity_label="manipulated")
    assert annot.binary_label == 1


def test_binary_label_unknown():
    annot = Annotation(authenticity_label="genuine")
    with pytest.raises(ValueError):
        _ = annot.binary_label


def get_label():
    annot = Annotation(authenticity_label="fake", source_label="model1")
    assert annot.get_label("authenticity_label") == "fake"
    assert annot.get_label("source_label") == "model1"
    assert annot.get_label("binary_label") == 1


def get_label_unknown_raises_error():
    annot = Annotation(authenticity_label="fake", source_label="model1")
    with pytest.raises(ValueError):
        _ = annot.get_label("unknown_label")
