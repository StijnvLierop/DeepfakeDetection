import functools
import operator
import pydoc
from typing import Callable, Union

import yaml

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.data.datasets import FilteredDataset, MappedDataset
from deepfake_detection.models import Model


OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "in": lambda x, s: x in s,
}


def filter_config_to_func(filter_config: dict) -> Callable:
    """
    Converts a filter config dict into a callable that accepts an Instance.

    Filters can be combined using logical operators 'and', 'or' and 'not'.

    :param filter_config: Filter configuration dictionary.
    :return: Function that implements the configured filter.
    """
    if "func" in filter_config:
        return func_config_to_func(filter_config)

    if "and" in filter_config:
        funcs = [filter_config_to_func(f) for f in filter_config["and"]]
        return lambda x: all(f(x) for f in funcs)

    if "or" in filter_config:
        funcs = [filter_config_to_func(f) for f in filter_config["or"]]
        return lambda x: any(f(x) for f in funcs)

    if "not" in filter_config:
        func = filter_config_to_func(filter_config["not"])
        return lambda x: not func(x)

    op = OPS[filter_config["op"]]

    def filter_func(instance):
        return op(
            instance.annotation.get_label(filter_config["label"]),
            filter_config["value"],
        )

    return filter_func


def func_config_to_func(config: dict) -> Callable:
    """
    This function takes a function configuration dictionary and returns a function that implements the configured function.

    The configuration should contain a 'func' key and optional 'params' specific for the function, e.g.:
    func: add_label
    params:
        label: 'example_label'
        value: 'example_value'

    :param config: Configuration dictionary.
    :return: Function that implements the configured function.
    """
    func = pydoc.locate(config["func"])
    if func is None:
        raise ValueError(f"Function {config['func']} not found.")
    return functools.partial(func, **config["params"])


def load_dataset(config: Union[str, dict]) -> Dataset:
    """
    Loads a dataset from a YAML file path or a config dict.

    :param config: The mapping should contain the classpath of the dataset to
                   use and the necessary parameters to initialize that dataset. E.g.

                   class: deepfake_detection.models.detection.FileImageDataset
                   params:
                       dataset_name: ExampleDataset
                       path: /path/to/dataset/
                       ...
    """
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)
    return build_dataset(config)


def build_dataset(config: dict) -> Dataset:
    """
    Instantiates a Dataset from a config dict, then applies map / filter / sample.

    The configuration should contain a 'class' key specifying the dataset class and optional 'params'
    specific for the dataset, e.g.:
    class: deepfake_detection.models.detection.FileImageDataset
    params:
        dataset_name: ExampleDataset
        path: /path/to/dataset/
        ...

    HuggingFace-specific options (hf_map, hf_filter, shuffle, take) belong inside
    'params' and are handled by HuggingfaceDataset.__init__ directly.
    """

    def _recursive_load(value):
        if isinstance(value, dict) and "class" in value:
            return build_dataset(value)
        if isinstance(value, dict):
            return {k: _recursive_load(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_recursive_load(v) for v in value]
        return value

    cls = pydoc.locate(config["class"])
    if cls is None:
        raise ValueError(f"Dataset class not found: {config['class']}")

    params = config.get("params", {}).copy()

    # Extract post-construction transformations before passing params to __init__
    map_cfg = params.pop("map", config.get("map", []))
    filter_cfg = params.pop("filter", config.get("filter"))
    sample_cfg = params.pop("sample", config.get("sample"))

    # Recursively build any nested dataset params
    params = {k: _recursive_load(v) for k, v in params.items()}

    # Initialize dataset object
    dataset = cls(**params)

    # Normalize map_cfg: support both a plain list and the legacy {funcs: [...]} dict form
    if isinstance(map_cfg, dict):
        map_cfg = map_cfg.get("funcs", [])

    # Apply map functions in order
    for func_cfg in map_cfg:
        dataset = MappedDataset(dataset, func_config_to_func(func_cfg))

    # Apply instance-level filter
    if filter_cfg:
        dataset = FilteredDataset(dataset, filter_config_to_func(filter_cfg))

    # Apply sampling
    if sample_cfg:
        dataset = func_config_to_func(sample_cfg)(dataset)

    return dataset


def load_model(config: Union[str, dict]) -> Model:
    """
    Loads a model from a YAML file path or a config dict.
    """
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)

    model_class = pydoc.locate(str(config["class"]))
    if model_class is None:
        raise ValueError(f"Model class not found: {config['class']}")
    return model_class(**config.get("params", {}))
