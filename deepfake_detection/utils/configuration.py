import confidence

from deepfake_detection.utils.parameters import DATASETS, MODELS


def parse_dataset_config(config_path: str):
    """
    This function parses the dataset configuration file and returns a dictionary of configured datasets.

    :param config_path: path to the configuration file.
    """
    
    config = confidence.loadf(config_path)
    datasets = {}

    for dataset in config.datasets:
        dataset_dict =  dict(dataset)
        type = dataset_dict.pop('type', None)
        datasets[dataset.name] = DATASETS[type](**dataset_dict)

    return datasets


def parse_model_config(config_path: str):
    """
    This function parses the models configuration file and returns a dictionary of configured models.

    :param config_path: path to the configuration file.
    """

    config = confidence.loadf(config_path)
    models = {}

    for model in config.models:
        model_args = dict(model)
        model_name = model_args.pop('name', None)
        models[model_name] = MODELS[model_name](**model_args)

    return models