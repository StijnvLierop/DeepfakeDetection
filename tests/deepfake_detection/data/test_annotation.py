import pytest

from deepfake_detection.data.annotation import Annotation


def test_get_label():
    annot = Annotation({'authenticity': "fake", 'source': "model1"})
    assert annot.get_label("authenticity") == "fake"
    assert annot.get_label("source") == "model1"


def test_get_label_unknown_raises_error():
    annot = Annotation({'authenticity': "fake", 'source': "model1"})
    with pytest.raises(ValueError):
        _ = annot.get_label("unknown_label")
