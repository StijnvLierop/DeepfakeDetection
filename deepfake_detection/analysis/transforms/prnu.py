import numpy as np

from deepfake_detection.analysis.prnu import prnu_fstv
from deepfake_detection.analysis.transforms.base import AnalysisTransform


class PRNUTransform(AnalysisTransform):
    """Wraps :func:`prnu_fstv` as a named, configurable transform."""

    @property
    def name(self) -> str:
        return "prnu"

    def apply(self, img: np.ndarray) -> np.ndarray:
        return prnu_fstv(img)
