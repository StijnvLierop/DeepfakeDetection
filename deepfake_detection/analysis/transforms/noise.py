import numpy as np

from deepfake_detection.analysis.noise import noise_residual
from deepfake_detection.analysis.transforms.base import AnalysisTransform


class NoiseResidualTransform(AnalysisTransform):
    """Wraps :func:`noise_residual` as a named, configurable transform."""

    def __init__(self, image_filter: str = "median"):
        self.image_filter = image_filter

    @property
    def name(self) -> str:
        return f"noise_residual_{self.image_filter}"

    def apply(self, img: np.ndarray) -> np.ndarray:
        return noise_residual(img, image_filter=self.image_filter)
