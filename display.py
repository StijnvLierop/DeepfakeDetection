import argparse
from typing import Optional

import fiftyone as fo

from deepfake_detection.data.datasets.fiftyone import to_fiftyone_dataset
from deepfake_detection.utils.configuration import load_dataset, load_model


def display(dataset: str, cache_dir: str, model: Optional[str] = None):
    # Load dataset
    dataset = load_dataset(dataset)

    # Load model (if specified)
    if model:
        model = load_model(model)

    # Convert to FiftyOne dataset
    fo_dataset = to_fiftyone_dataset(dataset, cache_dir=cache_dir, embedding_model=model)

    # Launch FiftyOne app
    session = fo.launch_app(fo_dataset)
    session.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        required=True,
                        help='Path to dataset config file of the dataset to display.')
    parser.add_argument('-c',
                        '--cache-dir',
                        type=str,
                        required=False,
                        help='Path to cache directory. Should be provided when displaying'
                             ' datasets that do not have individual sample files stored on disk.')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=False,
                        help='Path to model config file of the model to use for computing embeddings.')
    display(**vars(parser.parse_args()))