import pytest
from unittest.mock import MagicMock
from deepfake_detection.data.utils.map import map_label_values
from deepfake_detection.data.instance import Instance
from deepfake_detection.data.annotation import Annotation

def test_map_label_values_single_value():
    # Setup
    labels = {"label1": "old_value"}
    annotation = Annotation(labels)
    # Instance is abstract, so we mock it or use a simple mock-like object
    instance = MagicMock(spec=Instance)
    instance.annotation = annotation
    
    # Execute
    result = map_label_values(instance, "label1", "new_value")
    
    # Verify
    assert "new_value" == result.annotation.get_label("label1")
    assert instance == result

def test_map_label_values_dict_mapping():
    # Setup
    labels = {"label1": "old_value"}
    annotation = Annotation(labels)
    instance = MagicMock(spec=Instance)
    instance.annotation = annotation
    
    mapping = {"old_value": "mapped_value"}
    
    # Execute
    result = map_label_values(instance, "label1", mapping)
    
    # Verify
    assert "mapped_value" == result.annotation.get_label("label1")

def test_map_label_values_wildcard_mapping():
    # Setup
    labels = {"label1": "unknown_value"}
    annotation = Annotation(labels)
    instance = MagicMock(spec=Instance)
    instance.annotation = annotation
    
    mapping = {"known_value": "mapped_value", "*": "wildcard_value"}
    
    # Execute
    result = map_label_values(instance, "label1", mapping)
    
    # Verify
    assert "wildcard_value" == result.annotation.get_label("label1")
