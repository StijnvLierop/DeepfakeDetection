import os

import numpy as np
import pytest
from PIL import Image

from deepfake_detection.data import FileImageInstance, FileVideoInstance, FileImageSequenceInstance, ImageInstance, \
    Dataset
from deepfake_detection.data.annotation import Annotation
from deepfake_detection.data.datasets import ListDataset
from tests.deepfake_detection.paths import RESOURCES_DIR


@pytest.fixture
def file_image_instance():
    return FileImageInstance(path=os.path.join(RESOURCES_DIR, "data/test_image_dataset/fake/model1/fake01.png"),
                             annotation=Annotation(authenticity_label="fake", source_label="model1"))


@pytest.fixture
def image_instance():
    return ImageInstance(data=Image.open(os.path.join(RESOURCES_DIR,
                                                      "data/test_image_dataset/fake/model1/fake01.png")),
                         annotation=Annotation(authenticity_label="fake", source_label="model1"))


@pytest.fixture
def file_video_instance():
    return FileVideoInstance(path=os.path.join(RESOURCES_DIR, "data/test_image_dataset/fake/model1/fake1.mp4"),
                         annotation=Annotation(authenticity_label="fake", source_label="model1"))


@pytest.fixture
def file_image_sequence_instance():
    return FileImageSequenceInstance(path=os.path.join(RESOURCES_DIR,
                                                       "data/test_image_sequence_dataset/fake/model1/0_0fake"),
                                     annotation=Annotation(authenticity_label="fake", source_label="model1"))

@pytest.fixture
def image_dataset_path():
    return RESOURCES_DIR / "data" / "test_image_dataset"


@pytest.fixture
def image_sequence_dataset_path():
    return RESOURCES_DIR / "data" / "test_image_sequence_dataset"


@pytest.fixture
def video_dataset_path():
    return RESOURCES_DIR / "data" / "test_video_dataset"


@pytest.fixture
def dummy_dataset() -> Dataset:
    return ListDataset(
        instances=[ImageInstance(data=Image.fromarray(np.zeros(100+d)),
                                 annotation=Annotation(authenticity_label="real" if l == "A" else "fake",
                                                       source_label=l)
                                 ) for (d, l) in
                    zip(range(15),
                        ["A", "A", "B", "C", "C", "C", "B", "C", "C", "A", "A", "B", "C", "A", "B"])
                   ]
    )