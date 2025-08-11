import confidence

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.models import Model
from deepfake_detection.utils.parameters import DATASETS, MODELS


def parse_dataset_config(config_path: str):
    """
    This function parses the dataset configuration file and returns a dictionary of configured datasets.

    :param config_path: Path to the configuration file.
    """
    
    config = confidence.loadf(config_path)
    datasets = {}

    for dataset_config in config.datasets:
        datasets[dataset_config.name] = load_dataset(dataset_config)

    return datasets

def load_dataset(config: confidence.Configuration) -> Dataset:
    """
    This function initializes a dataset from a provided configuration mapping.

    :param config: Configuration mapping. The mapping should contain a name of the dataset present in
                   parameters.py and the necessary parameters to initialize that dataset.
    """
    dataset_dict = dict(config)
    type = dataset_dict.pop('type', None)
    return DATASETS[type](**dataset_dict)


def parse_model_config(config_path: str):
    """
    This function parses the models configuration file and returns a dictionary of configured models.

    :param config_path: Path to the configuration file.
    """

    config = confidence.loadf(config_path)
    models = {}

    for model_config in config.models:
        models[model_config.name] = load_model(model_config)

    return models


def load_model(config: confidence.Configuration) -> Model:
    """
    This function initializes a model from a provided configuration mapping.

    :param config: Configuration mapping. The mapping should contain a name of the model present in
                   parameters.py and the necessary parameters to initialize that model.
    """
    model_args = dict(config)
    model_name = model_args.pop('name', None)
    return MODELS[model_name](**model_args)