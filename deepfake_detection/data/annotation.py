from typing import Optional


class Annotation:
    """
    Represents an annotation for an instance. Each annotation has:
    - an authenticity label (fake, real or manipulated).
    - an optional source label containing the generator or camera used to generate the instance.
    """

    def __init__(self, authenticity_label: str, source_label: Optional[str] = None):
        """
        :param authenticity_label: authenticity label (fake, real or manipulated).
        :param source_label: optional source label containing the generator or camera used to generate the instance.
        """
        self.authenticity_label = authenticity_label
        self.source_label = source_label

    def get_label(self, label_type: str):
        """
        Convenience method to get the label of the annotation given a string.

        :param label_type: The type of label to return. Must be one of: 'authenticity_label', 'source_label'
                           or 'binary_label'.
        """
        if label_type == "authenticity_label":
            return self.authenticity_label
        elif label_type == "source_label":
            return self.source_label
        elif label_type == "binary_label":
            return self.binary_label
        else:
            raise ValueError(
                "Invalid label type. Must be one of: 'authenticity_label', "
                "'source_label' or 'binary_label'."
            )

    @property
    def binary_label(self) -> int:
        """
        Returns a binary integer label for the annotation.
        The binary label is 1 for fake or manipulated instances and 0 for real instances.
        """
        if self.authenticity_label is None:
            raise ValueError(
                "Authenticity label not set so cannot determine binary label."
            )
        if self.authenticity_label in ["fake", "manipulated"]:
            return 1
        elif self.authenticity_label == "real":
            return 0
        else:
            raise ValueError(
                "Invalid authenticity label. Must be one of: 'fake', 'real' or 'manipulated'."
            )
