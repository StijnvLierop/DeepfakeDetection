import math
from operator import itemgetter
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import numpy as np


class Prediction:
    """
    This class represents a single model prediction for an ``Instance``.

    :param classification: A mapping of textual labels to their corresponding
        confidence scores.
    :param embedding: A numerical embedding (which could be of any length),
        represented as a sequence of floats.
    :param text: A free-form text field, which can for example be used for
        image captioning, machine translation, summarization, etc.
    :param image: A matrix representing an image. To be used e.g. in
        segmentation tasks, where predictions are made at the pixel level.
    :param meta: A mapping where additional metadata about this prediction
        can be stored. Code should not depend on the presence of
        certain meta attributes, but they can be used to improve
        performance or provide additional functionality.
    """

    def __init__(
        self,
        classification: Mapping[str, float] = None,
        embedding: Sequence[float] = None,
        text: str = None,
        images: Mapping[str, np.ndarray] = None,
        meta: MutableMapping[str, Any] = None,
    ):
        #: A mapping of class labels to the corresponding confidence scores as
        #: predicted by the model
        self.classification: Mapping[str, float] = classification
        #: A numerical embedding (which could be of any length), represented as
        #: a sequence of floats.
        self.embedding: Sequence[float] = embedding
        #: A free-form text field, which can for example be used for image
        #: captioning, machine translation, summarization, etc.
        self.text: str = text
        #: One or multiple matrix annotations representing images.
        self.images: Mapping[str, np.ndarray] = images
        #: A dictionary that holds additional relevant metadata for a
        #: Prediction.
        self.meta: MutableMapping[str, Any] = meta or dict()

    @property
    def label(self) -> str:
        """
        Returns the single label with the highest classification confidence.
        """
        if not self.classification:
            raise ValueError("Can't determine label without classification")
        return max(self.classification.items(), key=itemgetter(1))[0]

    @property
    def confidence(self) -> float:
        """
        Returns the confidence score of the label with the
        highest classification confidence.
        """
        if not self.classification:
            raise ValueError("Can't return a confidence score without classification")
        return self.classification[self.label]

    def is_close(self, other: "Prediction", epsilon: float) -> bool:
        """Returns whether the other Prediction is close to the current
        Prediction with a given epsilon as the allowed margin

        :param other: The other Prediction to compare to
        :param epsilon: The margin within which Predictions are still
        considered close

        """
        return (
            self._compare_classification(other.classification, epsilon)
            and self._compare_embedding(other.embedding, epsilon)
            and self.text == other.text
            and all([np.array_equal(ri, oi) for ri, oi in zip(self.images.values(), other.images.values())])
            if (self.images is not None and other.images is not None)
            else self.images == other.images
        )

    def __hash__(self):
        return hash(
            (
                self.classification,
                self.embedding,
                self.text,
                str(self.images),
                str(self.meta),
            )
        )

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.is_close(other, epsilon=0.0)

    def __str__(self) -> str:
        return (
            f"Prediction("
            f"classification={self.classification}, "
            f"embedding={self.embedding}, "
            f"text={self.text}, "
            f"image={self.images.keys() if self.images is not None else None}, "
            f"meta={self.meta}"
            f")"
        )

    def __repr__(self) -> str:
        return str(self)

    def _compare_classification(
        self, classification: Optional[Mapping[str, float]], epsilon: float
    ) -> bool:
        if self.classification is None and classification is None:
            return True
        if self.classification is None or classification is None:
            return False
        if self.classification.keys() != classification.keys():
            return False
        for label, a in self.classification.items():
            b = classification[label]
            if epsilon and not math.isclose(a, b, abs_tol=epsilon):
                return False
            if not epsilon and a != b:
                return False
        return True

    def _compare_embedding(
        self, embedding: Optional[Sequence[float]], epsilon: float
    ) -> bool:
        if self.embedding is None and embedding is None:
            return True
        if self.embedding is None or embedding is None:
            return False
        if len(self.embedding) != len(embedding):
            return False
        return all(
            math.isclose(a, b, abs_tol=epsilon) if epsilon else a == b
            for a, b in zip(self.embedding, embedding)
        )
