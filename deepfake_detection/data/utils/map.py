from typing import Mapping, Union, List

from deepfake_detection.data.instance import FileImageInstance, ImageInstance
from deepfake_detection.data.annotation import Annotation
from deepfake_detection.data.instance import Instance


def convert_image_mode(instance: ImageInstance, mode: str) -> ImageInstance:
    """
    Converts the image data of an instance to the given PIL mode (e.g. ``"RGB"``, ``"L"``).
    Does not do anything when the image is already in the requested mode.

    :param instance: The image instance to convert.
    :param mode: Target PIL image mode.
    :return: The instance with the converted image.
    """
    if instance.data.mode != mode:
        instance.data = instance.data.convert(mode)
    return instance


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


def hf_parse_json_labels(
    row: dict,
    json_col: str,
    label_per_col: Mapping[str, str],
    drop_json_col: bool = True,
) -> dict:
    """
    Parses JSON labels from a HuggingFace dataset row and adds them as new columns.

    :param row: The HuggingFace dataset row to parse.
    :param json_col: The name of the JSON column in the row.
    :param label_per_col: A mapping of columns to labels in the json_col.
    :param drop_json_col: Whether to remove the original JSON column after parsing.
    """
    # Get/Parse JSON
    json_data = row.get(json_col, {})

    # Add new columns
    for json_key, new_col in label_per_col.items():
        row[new_col] = json_data.get(json_key)

    # Remove JSON column
    if drop_json_col:
        row.pop(json_col, None)

    return row


def hf_rename_column(row: dict, mapping: dict[str, str]) -> dict:
    """
    Renames columns of a HuggingFace dataset by modifying the row dictionary.

    :param row: The HuggingFace dataset row containing the columns to rename.
    :param mapping: {"old_col_name": "new_col_name"}
    """
    for old_name, new_name in mapping.items():
        if old_name in row:
            row[new_name] = row.pop(old_name)
    return row


def hf_map_label_values(
    row: dict, label: str, value: Union[str, Mapping[str, str]]
) -> dict:
    """
    This function maps the values of a particular label to a new value. This can be useful when renaming label values.

    :param row: The instance to remap the label value of.
    :param label: The label to remap the values of.
    :param value: The value to set. Can be a single value to replace a particular label value or a
                  mapping for more fine-grained changes.
                  If a mapping is provided, each value will be replaced by the
                  corresponding value in the mapping. A '*' functions as a wildcard and will be used
                  when no key is specified.
    :return: The instance with the remapped label value(s).
    """
    row[label] = value
    return row
