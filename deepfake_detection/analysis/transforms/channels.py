from typing import Sequence

import numpy as np

from deepfake_detection.analysis.channels import channel_threshold_map
from deepfake_detection.analysis.transforms.base import AnalysisTransform


class ChannelThresholdMap(AnalysisTransform):
    """Wraps :func:`channel_threshold_map` as a named, configurable transform."""

    def __init__(
        self,
        channels: Sequence[int],
        threshold: float,
        mode: str = "any",
        direction: str = "above",
    ):
        self.channels = list(channels)
        self.threshold = threshold
        self.mode = mode
        self.direction = direction

    @property
    def name(self) -> str:
        ch = "_".join(str(c) for c in self.channels)
        return f"channel_threshold_ch{ch}_t{int(self.threshold)}_{self.mode}_{self.direction}"

    def apply(self, img: np.ndarray) -> np.ndarray:
        return channel_threshold_map(img, self.channels, self.threshold, self.mode, self.direction)
