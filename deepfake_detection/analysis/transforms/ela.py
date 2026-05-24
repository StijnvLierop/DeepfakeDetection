import os
import tempfile

import numpy as np
from PIL import Image

from deepfake_detection.analysis.ela import ela
from deepfake_detection.analysis.transforms.base import AnalysisTransform


class ELATransform(AnalysisTransform):
    """
    Wraps :func:`ela` as a named transform.

    Because the underlying function requires a file path for JPEG recompression,
    the input array is saved to a temporary PNG before calling it.
    """

    @property
    def name(self) -> str:
        return "ela"

    def apply(self, img: np.ndarray) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            Image.fromarray(img).save(tmp_path)
            return np.array(ela(tmp_path))
        finally:
            os.unlink(tmp_path)
