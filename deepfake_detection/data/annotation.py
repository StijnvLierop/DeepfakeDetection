from typing import Mapping, Union, Optional


class Annotation:
    """
    Represents an annotation for an instance. Each annotation has one or multiple labels.
    """

    def __init__(self, labels: Mapping[str, str]):
        """
        :param labels: A mapping of label keys to their values.
        """
        self.labels = labels

    def __getitem__(self, item):
        return self.get_label(item)

    def __setitem__(self, key, value):
        return self.set_label(key, value)

    def get_label(self, label: str) -> str:
        """
        Convenience method to get the value of a particular label.

        :param label: The label to get the value of.
        """
        if label in self.labels:
            return self.labels[label]
        else:
            raise ValueError(f"Invalid label type. Must be one of {self.labels.keys()}")

    def set_label(self, label: str, value: Union[str, Mapping[str, str]]):
        """
        Convenience method to set the value of a particular label.

        :param label: The label to set the value of.
        :param value: The value to set. Can be a single value to replace a particular label value or a
                      mapping for more fine-grained changes.
                      If a mapping is provided, each value will be replaced by the
                      corresponding value in the mapping. A '*' functions as a wildcard and will be used
                      when no key is specified. If the label is not found in the mapping and no wildcard is
                      provided, the original value will be kept.
        """
        if isinstance(value, Mapping):
            if label not in self.labels:
                return
            current_value = self.labels[label]
            if current_value in value:
                self.labels[label] = value[current_value]
            elif '*' in value:
                self.labels[label] = value['*']
        else:
            self.labels[label] = value

    def __repr__(self) -> str:
        return str(self.labels)

    def copy(self):
        return Annotation(self.labels.copy())