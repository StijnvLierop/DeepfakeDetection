import argparse
from typing import Optional

import fiftyone as fo
import mimetypes

from deepfake_detection.data.datasets.fiftyone import to_fiftyone_dataset
from deepfake_detection.utils.configuration import load_dataset, load_model, load_transforms


mimetypes.add_type("image/webp", ".webp")


def display(
    dataset: str,
    cache_dir: str,
    model: Optional[str] = None,
    batch_size: int = 128,
    transforms: Optional[str] = None,
):
    """
    Display a given dataset in FiftyOne to allow for interactive exploration and analysis.

    :param dataset: Path to dataset config file of the dataset to display.
    :param cache_dir: Directory to cache dataset files.
    :param model: Optional path to embedding model.
    :param batch_size: Batch size for streaming samples into FiftyOne dataset.
    :param transforms: Optional path to transforms config YAML file.
    """
    # Load dataset
    dataset = load_dataset(dataset)

    # Load model if provided
    if model:
        model = load_model(model)

    # Load analysis transforms if provided
    analysis_transforms = load_transforms(transforms) if transforms else []

    # Delete any existing dataset with this name before launching the app.
    if fo.dataset_exists(dataset.dataset_name):
        fo.delete_dataset(dataset.dataset_name)

    # Launch the app immediately with an empty dataset so samples are visible
    # as they stream in rather than only after the full dataset is loaded
    fo_dataset = fo.Dataset(name=dataset.dataset_name)
    session = fo.launch_app(fo_dataset)

    # Stream samples into FiftyOne dataset
    extra = {"batch_size": batch_size} if batch_size is not None else {}
    to_fiftyone_dataset(
        dataset,
        cache_dir=cache_dir,
        embedding_model=model,
        fo_dataset=fo_dataset,
        transforms=analysis_transforms,
        **extra,
    )

    # Keep session alive
    session.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset config file of the dataset to display.",
    )
    parser.add_argument(
        "-c",
        "--cache-dir",
        type=str,
        required=False,
        help="Path to cache directory. Should be provided when displaying"
        " datasets that do not have individual sample files stored on disk.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        help="Path to model config file of the model to use for computing embeddings.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        required=False,
        help="Batch size for streaming samples into FiftyOne dataset.",
    )
    parser.add_argument(
        "-t",
        "--transforms",
        type=str,
        required=False,
        help="Path to transforms config YAML file.",
    )
    display(**vars(parser.parse_args()))
