import numpy as np
from PIL import Image

from deepfake_detection.analysis.ela import ela
from deepfake_detection.analysis.transforms.base import AnalysisTransform


class ELATransform(AnalysisTransform):
    """Wraps :func:`ela` as a named transform."""

    def __init__(self, quality: int = 95):
        self.quality = quality

    @property
    def name(self) -> str:
        return f"ela_q{self.quality}"

    def apply(self, img: np.ndarray) -> np.ndarray:
        return np.array(ela(Image.fromarray(img), self.quality))
