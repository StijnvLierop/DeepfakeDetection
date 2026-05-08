from unittest.mock import MagicMock

from deepfake_detection.data.annotation import Annotation
from deepfake_detection.data.instance import Instance
from deepfake_detection.data.utils.map import map_label_values


def test_map_label_values_single_value():
    # Setup instance
    labels = {"label1": "old_value"}
    annotation = Annotation(labels)
    instance = MagicMock(spec=Instance)
    instance.annotation = annotation

    # Remap labels
    result = map_label_values(instance, "label1", "new_value")

    # Check whether the label was remapped correctly
    assert "new_value" == result.annotation.get_label("label1")
    assert instance == result


def test_map_label_values_dict_mapping():
    # Setup instance
    labels = {"label1": "old_value", "label2": "other_value"}
    annotation = Annotation(labels)
    instance = MagicMock(spec=Instance)
    instance.annotation = annotation

    # Define mapping
    mapping = {"old_value": "mapped_value"}

    # Remap labels
    result = map_label_values(instance, "label1", mapping)

    # Check whether the label was remapped correctly
    assert "mapped_value" == result.annotation.get_label("label1")
    assert "other_value" == result.annotation.get_label("label2")


def test_map_label_values_wildcard_mapping():
    # Setup instance
    labels = {"label1": "unknown_value"}
    annotation = Annotation(labels)
    instance = MagicMock(spec=Instance)
    instance.annotation = annotation

    # Define mapping
    mapping = {"known_value": "mapped_value", "*": "wildcard_value"}

    # Perform remapping
    result = map_label_values(instance, "label1", mapping)

    # Check if wildcard remapping worked correctly
    assert "wildcard_value" == result.annotation.get_label("label1")


def test_map_label_new_label():
    # Setup instance
    labels = {"label1": "old_value"}
    annotation = Annotation(labels)
    instance = MagicMock(spec=Instance)
    instance.annotation = annotation

    # Remap labels
    result = map_label_values(instance, "label2", "label2_value")

    # Check whether the label was remapped correctly
    assert "label2_value" == result.annotation.get_label("label2")
    assert result.annotation.labels == {"label1": "old_value", "label2": "label2_value"}


def test_map_label_values_no_match_no_wildcard():
    # Setup instance
    labels = {"label1": "unknown_value"}
    annotation = Annotation(labels)
    instance = MagicMock(spec=Instance)
    instance.annotation = annotation

    # Define mapping
    mapping = {"known_value": "mapped_value"}

    # Execute and verify that the label remains unchanged if no match and no wildcard
    result = map_label_values(instance, "label1", mapping)
    assert result.annotation.get_label("label1") == "unknown_value"
