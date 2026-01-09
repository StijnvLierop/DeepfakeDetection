import pydoc
from functools import partial
from typing import Any, Iterable, MutableMapping

import confidence

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.models import Model


def parse_dataset_config(config_path: str):
    """
    This function parses the dataset configuration file and returns a dictionary of configured datasets.

    :param config_path: Path to the configuration file.
    """

    config = confidence.loadf(config_path)
    datasets = {}

    for dataset_config in config.datasets:
        datasets[dataset_config.params.name] = load_dataset(dataset_config)

    return datasets


def load_dataset(config: confidence.Configuration) -> Any:
    """
    This function initializes a dataset from a provided configuration mapping.

    :param config: Configuration mapping. The mapping should contain the classpath of the dataset to
                   use and the necessary parameters to initialize that dataset. E.g.

                   class: deepfake_detection.data.datasets.FileImageDataset
                   params:
                       name: TestDataset
                       path: /path/to/dataset/folder/
                       ...
    """

    # Handle primitives
    if isinstance(config, (str, int, float, bool)) or config is None:
        return config

    # If dict like object
    if isinstance(config, dict) or hasattr(config, "keys"):

        # Detect if we are dealing with a function (with parameters)
        if "func" in config:
            func_path = config["func"]
            func_params = config.get("params", {})

            # Look for the function
            target_func = pydoc.locate(func_path)
            if target_func is None:
                raise ImportError(f"Function not found: {func_path}")

            return partial(target_func, **func_params)

        # If a class key is present, we initialize the dataset class
        if "class" in config:

            # Load config as dict
            config = dict(config)

            # Look for the dataset class
            cls_path = config.pop("class")
            cls = pydoc.locate(str(cls_path))
            if cls is None:
                raise ImportError(f"Dataset class not found: {cls_path}")

            # Get function arguments
            init_args = config.pop("params") if "params" in config else config

            # Recursively process arguments
            processed_args = {k: load_dataset(v) for k, v in init_args.items()}
            print(processed_args)
            print(cls)
            return cls(**processed_args)

        return {k: load_dataset(v) for k, v in config.items()}

    # Handle iterables
    if isinstance(config, Iterable):
        return [load_dataset(item) for item in list(config)]

    else:
        return config


def parse_model_config(config_path: str):
    """
    This function parses the models configuration file and returns a dictionary of configured models.

    :param config_path: Path to the configuration file.
    """

    config = confidence.loadf(config_path)
    models = {}

    for model_config in config.models:
        models[model_config.params.name] = load_model(model_config)

    return models


def load_model(config: confidence.Configuration) -> Model:
    """
    This function initializes a model from a provided configuration mapping.

    :param config: Configuration mapping. The mapping should contain the classpath of the model to
                   use and the necessary parameters to initialize that model. E.g.

                   class: deepfake_detection.models.detection.CNNDetect
                   params:
                       name: CNNDetect
                       ckpt: /path/to/model/checkpoint/
                       ...
    """
    # Convert configuration to dictionary
    model_dict = dict(config)

    # Find model class specified in config
    model_class = pydoc.locate(str(model_dict["class"]))

    # If class found, initialize dataset
    if model_class is not None:
        return model_class(**model_dict["params"])
    else:
        raise ValueError(f"Dataset class not found: {model_dict['class']}")
