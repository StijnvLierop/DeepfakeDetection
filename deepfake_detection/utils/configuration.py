import functools
import operator
import pydoc
from typing import Callable, Union

import yaml

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.data.datasets import FilteredDataset, MappedDataset
from deepfake_detection.models import Model


# Define filter operants
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
    This function takes a filter configuration dictionary and returns a function that implements the configured filter.

    A filter can be a predefined filter function or a logical operator (similar to map).
    Alternatively, when a filter is a logical operator, it is always defined by three keys:
    - 'label': the label in annotation to look at when filtering.
    - 'op': the operator to use for filtering.
    - 'value': the value to filter on.

    Filters can be combined using logical operators 'and', 'or' and 'not'.

    :param filter_config: Filter configuration dictionary.
    :return: Function that implements the configured filter.
    """
    # If function is provided
    if "func" in filter_config:
        return func_config_to_func(filter_config)

    # Handle logical operators
    if "and" in filter_config:
        funcs = [filter_config_to_func(f) for f in filter_config["and"]]
        return lambda x: all(f(x) for f in funcs)

    if "or" in filter_config:
        funcs = [filter_config_to_func(f) for f in filter_config["or"]]
        return lambda x: any(f(x) for f in funcs)

    if "not" in filter_config:
        func = filter_config_to_func(filter_config["not"])
        return lambda x: not func(x)

    # Create filter function
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
    # Retrieve the configured function
    func = pydoc.locate(config["func"])
    if func is None:
        raise ValueError(f"Function {config['func']} not found.")

    # Partially initialize function with configured parameters
    func = functools.partial(func, **config["params"])

    return func


def load_dataset(config: Union[str, dict]) -> Dataset:
    """
    This function loads a dataset from a given .yaml file or configuration dictionary.

    :param config: The mapping should contain the classpath of the dataset to
                   use and the necessary parameters to initialize that dataset. E.g.

                   class: deepfake_detection.models.detection.FileImageDataset
                   params:
                       dataset_name: ExampleDataset
                       path: /path/to/dataset/
                       ...
    """
    # Read configuration if needed
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)
    return build_dataset(config)


def build_dataset(config: dict) -> Dataset:
    """
    This function initializes a dataset from a provided configuration mapping.

    The configuration should contain a 'class' key specifying the dataset class and optional 'params'
    specific for the dataset, e.g.:
    class: deepfake_detection.models.detection.FileImageDataset
    params:
        dataset_name: ExampleDataset
        path: /path/to/dataset/
        ...

    In addition, optional 'map', 'filter' and 'sample' configurations can be provided which will be
    applied to the dataset in this order.

    :param config: Configuration mapping.
    :return: Dataset instance.
    """

    def _recursive_load(value):
        if isinstance(value, dict):
            if "class" in value:
                return build_dataset(value)
            return {k: _recursive_load(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_recursive_load(v) for v in value]
        else:
            return value

    # Find the configured class
    cls = pydoc.locate(config["class"])

    # Find params
    params = config.get("params", {}).copy()

    # Extract optional mapping/filtering/sampling configuration
    filter_cfg = params.pop("filter", config.get("filter"))
    sample_cfg = params.pop("sample", config.get("sample"))
    map_cfg = params.pop("map", config.get("map", []))

    # If a param is another dataset, load this first recursively
    params = {k: _recursive_load(v) for k, v in params.items()}

    # Initialize Dataset class
    dataset = cls(**params)

    # If map config, wrap MappedDataset around Dataset
    if map_cfg:
        # Loop over configured functions
        for func in map_cfg:
            map_func = func_config_to_func(func)
            dataset = MappedDataset(dataset, map_func)

    # If filter config, wrap FilteredDataset around Dataset
    if filter_cfg:
        filter_func = filter_config_to_func(filter_cfg)
        dataset = FilteredDataset(dataset, filter_func)

    # If sample config, wrap FilteredDataset around Dataset
    if sample_cfg:
        sample_func = func_config_to_func(sample_cfg)
        dataset = sample_func(dataset)

    return dataset


def load_model(config: Union[str, dict]) -> Model:
    """
    This function loads a model from a given .yaml file or configuration dictionary.

    :param config: Configuration dictionary or path to a .yaml file.
                   The mapping should contain the classpath of the model to
                   use and the necessary parameters to initialize that model. E.g.

                   class: deepfake_detection.models.detection.CNNDetect
                   params:
                       name: CNNDetect
                       ckpt: /path/to/model/checkpoint/
                       ...
    """
    # Read configuration if needed
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)

    # Find model class specified in config
    model_class = pydoc.locate(str(config["class"]))

    # If class found, initialize dataset
    if model_class is not None:
        return model_class(**config.get("params", {}))
    else:
        raise ValueError(f"Model class not found: {config['class']}")
