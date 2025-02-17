from deepfake_detection.data.datasets.FileImageDataset import FileImageDataset
from deepfake_detection.data.datasets.FileVideoDataset import FileVideoDataset
from deepfake_detection.data.datasets.GenImageDataset import GenImageDataset
from deepfake_detection.models.detection.DIF_2020.model import DIFModel
from deepfake_detection.models.detection.corvi_2023.model import Corvi2023Model
from deepfake_detection.models.detection.cozzolino_2023.model import Cozzolino2023Model

# Define the possible dataset classes that can be initialized
DATASETS = {'FileImageDataset' : FileImageDataset,
            'FileVideoDataset' : FileVideoDataset,
            'GenImageDataset' : GenImageDataset}

# Define the possible model classes that can be initialized
MODELS = {'Cozzolino2023' : Cozzolino2023Model,
          'Corvi2023' : Corvi2023Model,
          'DIF' : DIFModel}