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


def test_annotation_set_label_direct_value():
    # Setup
    labels = {"label1": "old_value"}
    annotation = Annotation(labels)
    # Execute
    annotation.set_label("label1", "new_value")
    # Verify
    assert "new_value" == annotation.get_label("label1")


def test_annotation_set_label_mapping_exact_match():
    # Setup
    labels = {"label1": "old_value"}
    annotation = Annotation(labels)
    mapping = {"old_value": "new_value", "other": "ignored"}
    # Execute
    annotation.set_label("label1", mapping)
    # Verify
    assert "new_value" == annotation.get_label("label1")


def test_annotation_set_label_mapping_wildcard_match():
    # Setup
    labels = {"label1": "old_value"}
    annotation = Annotation(labels)
    mapping = {"other": "ignored", "*": "wildcard_value"}
    # Execute
    annotation.set_label("label1", mapping)
    # Verify
    assert "wildcard_value" == annotation.get_label("label1")


def test_annotation_setitem_calls_set_label():
    # Setup
    labels = {"label1": "old_value"}
    annotation = Annotation(labels)
    # Execute
    annotation["label1"] = "new_value"
    # Verify
    assert "new_value" == annotation.get_label("label1")
