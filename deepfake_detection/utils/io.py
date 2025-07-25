import json
import os
import warnings
from json import JSONDecoder
from typing import Sequence, Mapping, Collection, Any, Optional, Tuple, Dict

import numpy as np
import streamlit as st

from deepfake_detection.data.datasets import ListDataset
from deepfake_detection.data.dataset import Dataset
from deepfake_detection.models.prediction import Prediction


def write_predictions_to_file(results_dir: str,
                              predictions: Sequence[Prediction],
                              dataset: Dataset,
                              model_name: str) -> str:
    """
    This function writes a set of predictions corresponding to a given dataset to a json file.

    :param results_dir: The path to the directory where the predictions will be written to.
    :param predictions: The predictions to write to a file.
    :param dataset: The dataset corresponding to the predictions.
    :param model_name: The name of the model that made the predictions.
    :return: The path to the json file where the predictions are written.
    """

    # Ensure that the length of prediction and dataset is the same
    if len(predictions) != len(dataset):
        raise ValueError("Predictions must have the same length as the dataset!")

    # Encode predictions
    encoded_predictions = [encode_prediction(p) for p in predictions]

    # Write to file
    filename = os.path.join(results_dir, get_predictions_filename(dataset.name, model_name))
    with open(filename, 'w') as outfile:
        outfile.write(json.dumps(encoded_predictions))

    return filename

def encode_prediction(obj: Prediction) -> Dict[str, Any]:
    """
    Serializes a `Prediction` to JSON.

    :param obj: `Prediction` to serialize
    :return: JSON-encoded representation of the `Prediction` object
    """
    return {
        "classification": obj.classification,
        "embedding": obj.embedding,
        "text": obj.text,
        "image": obj.image.tolist() if (obj.image is not None) else None,
        "meta": encode_meta(obj.meta),
    }

@staticmethod
def encode_meta(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Takes arbitrary `meta` data and tries to serialize as many fields as
    possible. Any field that cannot be serialized is excluded from the
    result set.

    :param meta: The `meta` data to serialize
    :return: The serializable `meta` attributes
    """

    def encode_attribute(attr: str) -> Optional[Tuple[str, Any]]:
        try:
            return attr, jsonify(meta[attr])
        except TypeError:
            warnings.warn(
                f"Skipping meta attribute '{attr}' because it cannot be "
                f"serialized"
            )
            return None

    return dict(filter(None, map(encode_attribute, meta)))


def decode_prediction(obj: Mapping[str, Any]) -> Prediction:
    """
    Deserializes a JSON-encoded `Prediction`.

    :param obj: JSON-encoded representation of a `Prediction` object
    :return: Deserialized `Prediction` object
    """
    image = None if obj.get("image") is None else np.array(obj["image"])
    return Prediction(
        classification=obj["classification"],
        embedding=obj["embedding"],
        text=obj["text"],
        image=image,
        meta=obj.get("meta")
    )

@st.cache_data
def read_predictions_from_file(predictions_path: str) -> Sequence[Prediction]:
    """
    This function reads a set of predictions corresponding to a given dataset from a file.

    :param predictions_path: The path to the file where the predictions are stored.
    """
    with open(predictions_path, 'r') as infile:
        return [decode_prediction(p) for p in json.load(infile)]


def get_predictions_filename(dataset_name: str, model_name: str) -> str:
    """
    This function returns a filename for a new predictions file.

    :param dataset_name: The dataset name corresponding to the predictions.
    :param model_name: The name of the model that made the predictions.
    """
    return f'predictions_{dataset_name}_{model_name}.json'


def jsonify(obj: Any) -> Any:
    """
    Recursively breaks down an `obj` into simpler data-types that can easily be
    serialized to JSON.

    :param obj: Any object to be converted to JSON-serializable types
    :return: The JSON-serializable object
    """
    # Since `isinstance(str, Sequence)` we handle `str` here separately, since
    # we don't want to apply the default behaviour for sequences to `str`.
    if isinstance(obj, str):
        return obj

    # We generally don't want to serialize `bytes` and `bytearray`, so raise a
    # `TypeError` here before they are handled as a `Sequence`.
    if isinstance(obj, (bytes, bytearray)):
        raise TypeError(f"Can't jsonify object of type {type(obj).__name__}")

    # We recursively serialize all public items in a mapping.
    if isinstance(obj, Mapping):
        return {
            k: jsonify(v) for k, v in obj.items() if not k.startswith('_')
        }

    # We recursively serialize any items in a `Sequence`. Any cases where `obj`
    # is a `Sequence` but should not be handled as such (e.g. `str`) should
    # have been handled by now.
    if isinstance(obj, Collection):
        return list(map(jsonify, obj))

    # If we're dealing with an object, we serialize its attributes.
    if hasattr(obj, '__dict__'):
        return jsonify(vars(obj))

    # For all other cases, we check whether `json.dumps()` accepts `obj`.
    # If so, we simply return the object as is. If not, we raise a `TypeError`.
    # This follows the mantra of "fail early" and prevents an error from being
    # raised much later in the code, when it's much harder to track down the
    # root cause.
    try:
        json.dumps(obj)
        return obj
    except TypeError as e:
        message = f"Can't jsonify object of type {type(obj).__name__}: {e}"
        raise TypeError(message) from e


def load_dataset_from_file(dataset_path: str, decoder: JSONDecoder = None) -> ListDataset:
    """
    This function loads a dataset from a .JSON file.

    :param dataset_path: The path to the .JSON file containing the instances.
    :param decoder: The decoder to use for decoding the instances.
    """
    # Read from file
    with open(dataset_path, 'r') as infile:

        # Decode instances
        instances = json.load(infile, cls=decoder)

    return ListDataset(instances=instances)