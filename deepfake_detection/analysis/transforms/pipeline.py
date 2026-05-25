import numpy as np

from deepfake_detection.analysis.transforms.base import AnalysisTransform


class TransformPipeline(AnalysisTransform):
    """
    Chains multiple transforms sequentially. The output of each transform
    is passed as input to the next.
    """

    def __init__(self, *transforms: AnalysisTransform):
        if not transforms:
            raise ValueError("TransformPipeline requires at least one transform.")
        self.transforms = transforms

    @property
    def name(self) -> str:
        return "__".join(t.name for t in self.transforms)

    def apply(self, img: np.ndarray) -> np.ndarray:
        result = img
        for t in self.transforms:
            result = t.apply(result)
        return result
