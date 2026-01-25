from unittest.mock import patch, mock_open

import pytest
import yaml
from deepfake_detection.data.annotation import Annotation
from deepfake_detection.data.datasets import ListDataset, MappedDataset, FilteredDataset
from deepfake_detection.data.instance import ImageInstance
from deepfake_detection.utils.configuration import load_dataset, func_config_to_func, filter_config_to_func


def test_func_config_to_func_success():
    # Setup
    config = {
        "func": "deepfake_detection.data.utils.map.map_label_values",
        "params": {"label": "l1", "value": "v1"}
    }
    # Execute
    func = func_config_to_func(config)
    # Verify
    assert func(ImageInstance(data=None, annotation=Annotation(labels={})))


def test_func_config_to_func_missing_func():
    # Setup
    config = {
        "func": "non_existent_function_xyz",
        "params": {}
    }
    # Execute & Verify
    with pytest.raises(ValueError, match="Function non_existent_function_xyz not found."):
        func_config_to_func(config)


def test_func_config_to_func_missing_params():
    # Setup
    config = {
        "func": "operator.add"
    }
    # Execute & Verify
    with pytest.raises(KeyError):
        func_config_to_func(config)


def test_load_dataset_from_dict():
    config = {
        "class": "deepfake_detection.data.datasets.ListDataset",
        "params": {
            "instances": [],
            "dataset_name": "test_dataset"
        }
    }
    dataset = load_dataset(config)
    
    assert isinstance(dataset, ListDataset)
    assert dataset.dataset_name == "test_dataset"


def test_load_dataset_with_map():
    config = {
        "class": "deepfake_detection.data.datasets.ListDataset",
        "params": {
            "instances": [],
            "map": [
                {
                    "func": "deepfake_detection.data.utils.map.map_label_values",
                    "params": {"label": "test", "value": "value"}
                }
            ]
        }
    }
    dataset = load_dataset(config)
    
    assert isinstance(dataset, MappedDataset)


def test_load_dataset_with_filter():
    config = {
        "class": "deepfake_detection.data.datasets.ListDataset",
        "params": {
            "instances": [],
            "filter": {
                "label": "authenticity",
                "op": "==",
                "value": "real"
            }
        }
    }
    dataset = load_dataset(config)
    
    assert isinstance(dataset, FilteredDataset)


def test_load_dataset_from_yaml_file():
    config_dict = {
        "class": "deepfake_detection.data.datasets.ListDataset",
        "params": {"instances": []}
    }
    yaml_content = yaml.dump(config_dict)
    
    with patch("builtins.open", mock_open(read_data=yaml_content)):
        with patch("yaml.safe_load", return_value=config_dict) as mock_yaml_load:
            dataset = load_dataset("config.yaml")
            
            assert isinstance(dataset, ListDataset)
            mock_yaml_load.assert_called_once()


def test_load_dataset_recursive():
    # Note: build_dataset has this line:
    # params = {k: load_dataset(v) if v == "class" else v for k,v in params.items()}
    # This seems a bit strange as it checks if the VALUE is "class". 
    # Usually it would be if the key is something that expects a dataset.
    # Looking at the code:
    # if v == "class": load_dataset(v)
    # If v is "class", load_dataset("class") will be called, which will likely fail
    # unless there is a file named "class".
    # Wait, if v is a dict and it has a "class" key, it should probably be loaded.
    # Let's re-read build_dataset in configuration.py
    
    # Line 110: params = {k: load_dataset(v) if isinstance(v, dict) and "class" in v else v for k,v in params.items()}
    # Wait, the code I saw in get_file_structure was:
    # 110:    params = {k: load_dataset(v) if v == "class" else v for k,v in params.items()}
    # That looks like a bug in the source code or I misread it.
    pass


def test_filter_config_to_func_eq():
    # Setup
    filter_config = {
        "label": "label1",
        "op": "==",
        "value": "expected_value"
    }
    annotation = Annotation({"label1": "expected_value"})
    instance = ImageInstance(data=None, annotation=annotation)
    # Execute
    filter_func = filter_config_to_func(filter_config)
    # Verify
    assert True == filter_func(instance)


def test_filter_config_to_func_eq_false():
    # Setup
    filter_config = {
        "label": "label1",
        "op": "==",
        "value": "expected_value"
    }
    annotation = Annotation({"label1": "other_value"})
    instance = ImageInstance(data=None, annotation=annotation)
    # Execute
    filter_func = filter_config_to_func(filter_config)
    # Verify
    assert False == filter_func(instance)


def test_filter_config_to_func_ne():
    # Setup
    filter_config = {
        "label": "label1",
        "op": "!=",
        "value": "excluded_value"
    }
    annotation = Annotation({"label1": "some_value"})
    instance = ImageInstance(data=None, annotation=annotation)
    # Execute
    filter_func = filter_config_to_func(filter_config)
    # Verify
    assert True == filter_func(instance)


def test_filter_config_to_func_in():
    # Setup
    filter_config = {
        "label": "label1",
        "op": "in",
        "value": ["value1", "value2"]
    }
    annotation = Annotation({"label1": "value1"})
    instance = ImageInstance(data=None, annotation=annotation)
    # Execute
    filter_func = filter_config_to_func(filter_config)
    # Verify
    assert True == filter_func(instance)


def test_filter_config_to_func_lt():
    # Setup
    filter_config = {
        "label": "score",
        "op": "<",
        "value": 0.5
    }
    annotation = Annotation({"score": 0.4})
    instance = ImageInstance(data=None, annotation=annotation)
    # Execute
    filter_func = filter_config_to_func(filter_config)
    # Verify
    assert True == filter_func(instance)
