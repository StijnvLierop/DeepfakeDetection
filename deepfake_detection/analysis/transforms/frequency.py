import numpy as np

from deepfake_detection.analysis.frequency import fft
from deepfake_detection.analysis.transforms.base import AnalysisTransform


class FFTTransform(AnalysisTransform):
    """Wraps :func:`fft` as a named, configurable transform."""

    def __init__(self, hamming_window: bool = False):
        self.hamming_window = hamming_window

    @property
    def name(self) -> str:
        return "fft_hamming" if self.hamming_window else "fft"

    def apply(self, img: np.ndarray) -> np.ndarray:
        return fft(img.copy(), hamming_window=self.hamming_window)
