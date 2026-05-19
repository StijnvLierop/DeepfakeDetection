from deepfake_detection.models.detection.cnnspot import CNNSpot
from deepfake_detection.models.detection.fatformer import FatFormer
from deepfake_detection.models.detection.freqnet import FreqNet
from deepfake_detection.models.detection.latte import Latte
from deepfake_detection.models.detection.npr import NPR
from deepfake_detection.models.detection.univfd import UnivFD
from deepfake_detection.models.detection.aide import AIDE
from deepfake_detection.models.detection.rppg import rPPGLSTM


__all__ = [
    "CNNSpot",
    "FreqNet",
    "Latte",
    "NPR",
    "UnivFD",
    "FatFormer",
    "AIDE",
    "rPPGLSTM",
]
