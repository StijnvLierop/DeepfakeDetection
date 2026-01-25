from typing import Mapping, Union

from deepfake_detection.data.instance import Instance


def map_label_values(instance: Instance, label: str, value: Union[str, Mapping[str, str]]):
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

