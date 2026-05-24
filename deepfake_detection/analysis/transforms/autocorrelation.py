import numpy as np

from deepfake_detection.analysis.autocorrelation import autocorrelation
from deepfake_detection.analysis.transforms.base import AnalysisTransform


class AutocorrelationTransform(AnalysisTransform):
    """
    Wraps :func:`autocorrelation` as a named, configurable transform.

    If a colour image is provided the channels are averaged to grayscale
    before computing autocorrelation, because the underlying function
    requires a 2-D input.
    """

    @property
    def name(self) -> str:
        return "autocorrelation"

    def apply(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = img.mean(axis=-1)
        return autocorrelation(img)
