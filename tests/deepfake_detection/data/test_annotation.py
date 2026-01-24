import pytest

from deepfake_detection.data.annotation import Annotation


def get_label():
    annot = Annotation(authenticity_label="fake", source_label="model1")
    assert annot.get_label("authenticity_label") == "fake"
    assert annot.get_label("source_label") == "model1"
    assert annot.get_label("binary_label") == 1


def get_label_unknown_raises_error():
    annot = Annotation(authenticity_label="fake", source_label="model1")
    with pytest.raises(ValueError):
        _ = annot.get_label("unknown_label")
