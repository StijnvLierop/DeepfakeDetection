from deepfake_detection.data.datasets.FileImageDataset import FileImageDataset
from deepfake_detection.models.detection.corvi_2023.model import Corvi2023Model
from deepfake_detection.models.detection.cozzolino_2023.model import Cozzolino2023Model

# Define the possible dataset classes that can be initialized
DATASETS = {'FileImageDataset' : FileImageDataset,}

# Define the possible model classes that can be initialized
MODELS = {'Cozzolino2023' : Cozzolino2023Model,
          'Corvi2023' : Corvi2023Model}