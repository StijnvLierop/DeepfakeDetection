import argparse
import tempfile
from typing import Optional

import fiftyone as fo
import mimetypes

from deepfake_detection.data.datasets.fiftyone import to_fiftyone_dataset
from deepfake_detection.utils.configuration import load_dataset, load_model


mimetypes.add_type("image/webp", ".webp")


def display(
    dataset: str, cache_dir: Optional[str] = None, model: Optional[str] = None, batch_size: int = 128
):
    """
    Display a given dataset in FiftyOne to allow for interactive exploration and analysis.

    :param dataset: Path to dataset config file of the dataset to display.
    :param cache_dir: Directory to cache dataset files. When omitted, a temporary
                      directory is created automatically and cleaned up after the
                      session ends. Required for datasets whose images are not
                      already stored as individual files on disk (e.g. HuggingFace).
    :param model: Optional path to embedding model.
    :param batch_size: Batch size for streaming samples into FiftyOne dataset.
    """
    # Load dataset
    dataset = load_dataset(dataset)

    # Load model if provided
    if model:
        model = load_model(model)

    # Delete any existing dataset with this name before launching the app.
    if fo.dataset_exists(dataset.dataset_name):
        fo.delete_dataset(dataset.dataset_name)

    # Launch the app immediately with an empty dataset so samples are visible
    # as they stream in rather than only after the full dataset is loaded
    fo_dataset = fo.Dataset(name=dataset.dataset_name)
    session = fo.launch_app(fo_dataset)

    # Use an auto-created temp dir when no cache_dir is given
    with tempfile.TemporaryDirectory() as tmp_dir:
        resolved_cache_dir = cache_dir or tmp_dir

        # Stream samples into FiftyOne dataset
        to_fiftyone_dataset(
            dataset,
            cache_dir=resolved_cache_dir,
            embedding_model=model,
            fo_dataset=fo_dataset,
            batch_size=batch_size,
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
        required=False,
        default=128,
        help="Batch size for streaming samples into FiftyOne dataset.",
    )
    display(**vars(parser.parse_args()))
