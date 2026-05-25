from abc import ABC, abstractmethod

import numpy as np


class AnalysisTransform(ABC):
    """
    Base class for analysis transforms. Each transform wraps a signal analysis
    function with a fixed configuration, giving it a name and a standardized interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """A valid Python identifier that uniquely describes this configured transform."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to a (H, W, C) or (H, W) image array.

        :param img: Input image as a numpy array.
        :return: Transformed image array.
        """
        raise NotImplementedError
