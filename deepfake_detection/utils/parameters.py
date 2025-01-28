from deepfake_detection.data.datasets.FileImageDataset import FileImageDataset
from deepfake_detection.models.detection.cozzolino_ea.model import Cozzolino2023Model


DATASETS = {
    'Synthbuster' : FileImageDataset(path='/mnt/extern/DeepFake/Datasets/Cozzolino_ea_2023/synthbuster',
                                     name='Synthbuster'),
    'CIFAKE_test' : FileImageDataset(path='/mnt/extern/DeepFake/Datasets/CIFAKE/test', name='CIFAKE_test')
}


MODELS = {
    'Cozzolino2023' : Cozzolino2023Model()
}