import pydoc
from typing import Any, Iterable

import confidence

from deepfake_detection.models import Model


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

            # Load config as dict
            config = dict(config)

            # Look for the function
            func_path = config.pop("func")
            target_func = pydoc.locate(func_path)
            if target_func is None:
                raise ImportError(f"Function not found: {func_path}")

            # Get function arguments
            init_args = config.pop("params") if "params" in config else config

            # Recursively process arguments
            processed_args = {k: load_dataset(v) for k, v in init_args.items()}

            return target_func(**processed_args)

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
            return cls(**processed_args)

        return {k: load_dataset(v) for k, v in config.items()}

    # Handle iterables
    if isinstance(config, Iterable):
        return [load_dataset(item) for item in list(config)]

    else:
        return config


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
