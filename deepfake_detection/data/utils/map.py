from typing import Mapping, Union, List

from deepfake_detection.data.instance import FileImageInstance
from deepfake_detection.data.annotation import Annotation
from deepfake_detection.data.instance import Instance


def map_label_values(
    instance: Instance, label: str, value: Union[str, Mapping[str, str]]
) -> Instance:
    """
    This function maps the values of a particular label to a new value. This can be useful when renaming label values.

    :param instance: The instance to remap the label value of.
    :param label: The label to remap the values of.
    :param value: The value to set. Can be a single value to replace a particular label value or a
                  mapping for more fine-grained changes.
                  If a mapping is provided, each value will be replaced by the
                  corresponding value in the mapping. A '*' functions as a wildcard and will be used
                  when no key is specified.
    :return: The instance with the remapped label value(s).
    """
    instance.annotation.set_label(label, value)
    return instance


def get_label_from_filename(
    instance: FileImageInstance, label: str, label_values: List[str]
) -> FileImageInstance:
    """
    This function extracts a label from the filename of an image instance and sets it as the annotation for the instance.
    If multiple label values are provided, the first matching value will be used. If no matching value is found,
    no annotation is added.

    :param instance: The image instance to process.
    :param label: The label of which the value is in the filename.
    :param label_values: A list of possible label values to search for in the filename.
    :return: The image instance with the extracted label set as annotation.
    """
    for value in label_values:
        if value in instance.path.name:
            instance.annotation = Annotation({label: value})
    return instance
