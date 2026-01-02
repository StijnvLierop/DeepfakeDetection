from deepfake_detection.data.instance import (Instance, FileImageInstance, ImageInstance,
                                              FileImageSequenceInstance, FileVideoInstance)
from deepfake_detection.data.dataset import Dataset, MapStyleDatasetMixin
from deepfake_detection.data.datasets.split_dataset import split_dataset
from deepfake_detection.data.utils import sample_n_per_class