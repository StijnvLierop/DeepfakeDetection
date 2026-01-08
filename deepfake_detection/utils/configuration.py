import pydoc

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


def load_dataset(config: confidence.Configuration) -> Dataset:
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
    # Convert configuration to dictionary
    dataset_dict = dict(config)

    # Find dataset class specified in config
    dataset_class = pydoc.locate(str(dataset_dict["class"]))

    # If class found, initialize dataset
    if dataset_class is not None:
        return dataset_class(**dataset_dict["params"])
    else:
        raise ValueError(f"Dataset class not found: {dataset_dict['class']}")


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
