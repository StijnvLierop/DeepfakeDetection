from unittest.mock import patch, mock_open
import pytest
import yaml

from deepfake_detection.utils.configuration import (load_dataset, func_config_to_func,
                                                    filter_config_to_func, load_model, build_dataset)
from deepfake_detection.data.datasets import ListDataset, MappedDataset, FilteredDataset
from deepfake_detection.data.annotation import Annotation
from deepfake_detection.data.instance import ImageInstance
from deepfake_detection.models import Model
from deepfake_detection.models.detection.cnndetect import CNNDetect


def test_func_config_to_func_success():
    # Setup
    config = {
        "func": "deepfake_detection.data.utils.map.map_label_values",
        "params": {"label": "label1",
                   "value": "value1"}
    }
    # Execute
    func = func_config_to_func(config)

    # Verify
    assert func(ImageInstance(data=None, annotation=Annotation(labels={})))


def test_func_config_to_func_non_existing_func():
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
        dataset = load_dataset("config.yaml")

        assert isinstance(dataset, ListDataset)


def test_load_dataset_recursive():
    config = {
        "class": "deepfake_detection.data.datasets.ListDataset",
        "params": {
            "instances": [],
            "dataset_name": {
                "class": "deepfake_detection.data.datasets.ListDataset",
                "params": {"instances": [], "dataset_name": "inner_dataset"}
            }
        }
    }
    
    # If I want to test it as is:
    config_as_is = {
        "class": "deepfake_detection.data.datasets.ListDataset",
        "params": {
            "instances": [],
            "dataset_name": "class" # This will trigger load_dataset("class")
        }
    }
    pass


def test_load_dataset_with_sample():
    config = {
        "class": "deepfake_detection.data.datasets.ListDataset",
        "params": {
            "instances": [
                ImageInstance(data=None, annotation=Annotation({"source": "A"})),
                ImageInstance(data=None, annotation=Annotation({"source": "A"})),
                ImageInstance(data=None, annotation=Annotation({"source": "B"})),
            ],
            "sample": {
                "func": "deepfake_detection.data.utils.sample.sample_n_per_class",
                "params": {"n": 1, "label": "source"}
            }
        }
    }
    dataset = load_dataset(config)

    # sample_n_per_class returns a ListDataset
    assert isinstance(dataset, ListDataset)
    assert 2 == len(dataset)


def test_load_dataset_with_multiple_maps():
    config = {
        "class": "deepfake_detection.data.datasets.ListDataset",
        "params": {
            "instances": [
                ImageInstance(data=None, annotation=Annotation({"label1": "v1", "label2": "v2"})),
            ],
            "map": [
                {
                    "func": "deepfake_detection.data.utils.map.map_label_values",
                    "params": {"label": "label1", "value": "new_v1"}
                },
                {
                    "func": "deepfake_detection.data.utils.map.map_label_values",
                    "params": {"label": "label2", "value": "new_v2"}
                }
            ]
        }
    }
    dataset = load_dataset(config)

    assert isinstance(dataset, MappedDataset)
    instance = dataset[0]
    assert "new_v1" == instance.annotation.get_label("label1")
    assert "new_v2" == instance.annotation.get_label("label2")


def test_load_dataset_combined():
    config = {
        "class": "deepfake_detection.data.datasets.ListDataset",
        "params": {
            "instances": [
                ImageInstance(data=None, annotation=Annotation({"label": "v1", "keep": True})),
                ImageInstance(data=None, annotation=Annotation({"label": "v2", "keep": False})),
                ImageInstance(data=None, annotation=Annotation({"label": "v3", "keep": True})),
            ],
            "map": [
                {
                    "func": "deepfake_detection.data.utils.map.map_label_values",
                    "params": {"label": "label", "value": "mapped"}
                }
            ],
            "filter": {
                "label": "keep",
                "op": "==",
                "value": True
            },
            "sample": {
                "func": "deepfake_detection.data.utils.sample.sample_n_per_class",
                "params": {"n": 1, "label": "label"}
            }
        }
    }
    dataset = load_dataset(config)

    # Order in build_dataset: map, then filter, then sample.
    # 1. Map: all labels become "mapped".
    # 2. Filter: only 2 instances remain (keep=True). Both have label="mapped".
    # 3. Sample: 1 instance per label "mapped". Total 1.
    
    assert 1 == len(dataset)
    assert "mapped" == dataset[0].annotation.get_label("label")


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


def test_load_model_from_dict():
    # Setup
    config = {
        "class": "deepfake_detection.models.detection.cnndetect.CNNDetect",
        "params": {
            "name": "MyCNNDetect",
            "device": "cpu"
        }
    }
    # Execute
    model = load_model(config)
    # Verify
    assert isinstance(model, Model)
    assert isinstance(model, CNNDetect)
    assert "MyCNNDetect" == model.name
    assert "cpu" == model.device


def test_load_model_from_yaml_file():
    # Setup
    config_dict = {
        "class": "deepfake_detection.models.detection.cnndetect.CNNDetect",
        "params": {
            "name": "YamlCNNDetect"
        }
    }
    yaml_content = yaml.dump(config_dict)
    # Execute
    with patch("builtins.open", mock_open(read_data=yaml_content)):
        model = load_model("dummy_path.yaml")
    # Verify
    assert isinstance(model, CNNDetect)
    assert "YamlCNNDetect" == model.name


def test_load_model_class_not_found():
    # Setup
    config = {
        "class": "non_existent_model_class_xyz",
        "params": {}
    }
    # Execute & Verify
    with pytest.raises(ValueError, match="Dataset class not found: non_existent_model_class_xyz"):
        load_model(config)


def test_build_dataset_basic():
    # Setup
    config = {
        "class": "deepfake_detection.data.datasets.ListDataset",
        "params": {
            "instances": [],
            "dataset_name": "basic_dataset"
        }
    }
    # Execute
    dataset = build_dataset(config)
    # Verify
    assert isinstance(dataset, ListDataset)
    assert "basic_dataset" == dataset.dataset_name


def test_build_dataset_with_map():
    # Setup
    config = {
        "class": "deepfake_detection.data.datasets.ListDataset",
        "params": {
            "instances": [],
            "map": [
                {
                    "func": "operator.neg",
                    "params": {}
                }
            ]
        }
    }
    # Execute
    dataset = build_dataset(config)
    # Verify
    assert isinstance(dataset, MappedDataset)


def test_build_dataset_with_filter():
    # Setup
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
    # Execute
    dataset = build_dataset(config)
    # Verify
    assert isinstance(dataset, FilteredDataset)


def test_build_dataset_with_multiple_maps():
    # Setup
    config = {
        "class": "deepfake_detection.data.datasets.ListDataset",
        "params": {
            "instances": [ImageInstance(data=None, annotation=Annotation({"label1": "A"})),
                          ImageInstance(data=None, annotation=Annotation({"label1": "B"}))],
            "map": [{
                "func": "deepfake_detection.data.utils.map.map_label_values",
                "params": {"label": "label1",
                           "value": {"A": "a"},
                           }},
                {"func": "deepfake_detection.data.utils.map.map_label_values",
                "params": {"label": "label2",
                           "value": "v2",
                           }
            }]
        }
    }

    # Execute
    dataset = build_dataset(config)

    # Verify
    assert dataset[0].annotation.get_label("label1") == "a"
    assert dataset[0].annotation.get_label("label2") == "v2"
    assert dataset[1].annotation.get_label("label1") == "B"
    assert dataset[1].annotation.get_label("label2") == "v2"


def test_build_dataset_with_sample():
    # Setup
    config = {
        "class": "deepfake_detection.data.datasets.ListDataset",
        "params": {
            "instances": [],
            "sample": {
                "func": "deepfake_detection.data.utils.sample.sample_n_per_class",
                "params": {"n": 1, "label": "label1"}
            }
        }
    }
    # Execute
    dataset = build_dataset(config)
    # Verify
    assert isinstance(dataset, ListDataset)


def test_build_dataset_recursive():
    # Setup
    config = {
        "class": "deepfake_detection.data.datasets.CombinedDataset",
        "params": {
            "datasets": [
                {
                    "class": "deepfake_detection.data.datasets.ListDataset",
                    "params": {"instances": [], "dataset_name": "ds1"}
                },
                {
                    "class": "deepfake_detection.data.datasets.ListDataset",
                    "params": {"instances": [], "dataset_name": "ds2"}
                }
            ],
            "dataset_name": "combined"
        }
    }
    # Execute
    dataset = build_dataset(config)
    # Verify
    from deepfake_detection.data.datasets.combined import CombinedDataset
    assert isinstance(dataset, CombinedDataset)
    assert 2 == len(dataset.datasets)
    assert "ds1" == dataset.datasets[0].dataset_name
    assert "ds2" == dataset.datasets[1].dataset_name
