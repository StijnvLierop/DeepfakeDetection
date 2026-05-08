from json import JSONEncoder, JSONDecoder
from typing import Union

import numpy as np

from deepfake_detection.data.instance import (
    ImageInstance,
    FileImageSequenceInstance,
    FileVideoInstance,
    FileImageInstance,
)
from deepfake_detection.data.annotation import Annotation


def serialize_annotation(annotation: Annotation):
    return {label: annotation.labels[label] for label in annotation.labels}


def deserialize_annotation(annotation_dict: dict):
    return Annotation(annotation_dict)


def serialize_file_instance(
    instance: Union[FileImageInstance, FileVideoInstance, FileImageSequenceInstance],
):
    return {
        "path": str(instance.path),
        "annotation": serialize_annotation(instance.annotation),
    }


def deserialize_file_instance(instance_dict: dict):
    return FileImageInstance(
        path=instance_dict["path"], annotation=instance_dict["annotation"]
    )


def serialize_image_instance(instance: ImageInstance):
    return {
        "data": np.array(instance.data).tolist(),
        "annotation": serialize_annotation(instance.annotation),
    }


def deserialize_image_instance(instance_dict: dict):
    return ImageInstance(
        data=instance_dict["data"], annotation=instance_dict["annotation"]
    )


class InstanceEncoder(JSONEncoder):
    """
    Class that encodes FileInstances into a JSON serializable format.
    Only metadata is encoded (except for ImageInstance), but data itself is not serialized.
    """

    def default(self, input):
        if isinstance(
            input,
            Union[FileImageInstance, FileVideoInstance, FileImageSequenceInstance],
        ):
            return serialize_file_instance(input)
        elif isinstance(input, ImageInstance):
            return serialize_image_instance(input)
        elif isinstance(input, Annotation):
            serialize_annotation(input)


class InstanceDecoder(JSONDecoder):
    """
    Class that decodes JSON data into an Instance.
    Only metadata is decoded (except for ImageInstance), but data itself is not stored in JSON format.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.entry_object_hook, *args, **kwargs)

    def entry_object_hook(self, obj):
        if "path" in obj:
            return deserialize_file_instance(obj)
        elif "annotation" in obj:
            return deserialize_annotation(obj)
        elif "data" in obj:
            return deserialize_image_instance(obj)
