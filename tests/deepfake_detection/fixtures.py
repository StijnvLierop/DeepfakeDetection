import os

import pytest
from PIL import Image

from deepfake_detection.data import FileImageInstance, FileVideoInstance, FileImageSequenceInstance, ImageInstance
from tests.deepfake_detection.paths import RESOURCES_DIR

@pytest.fixture
def file_image_instance():
    return FileImageInstance(path=os.path.join(RESOURCES_DIR, "data/test_image_dataset/fake/model1/fake01.png"),
                             label="fake")

@pytest.fixture
def image_instance():
    return ImageInstance(data=Image.open(os.path.join(RESOURCES_DIR,
                                                      "data/test_image_dataset/fake/model1/fake01.png")),
                         label="fake")


@pytest.fixture
def file_video_instance():
    return FileVideoInstance(path=os.path.join(RESOURCES_DIR, "data/test_image_dataset/fake/model1/fake1.mp4"),
                         label="fake")


@pytest.fixture
def file_image_sequence_instance():
    return FileImageSequenceInstance(path=os.path.join(RESOURCES_DIR,
                                                       "data/test_image_sequence_dataset/fake/model1/0_0fake"),
                                     label="real")

@pytest.fixture
def image_dataset_path():
    return RESOURCES_DIR / "data" / "test_image_dataset"


@pytest.fixture
def image_sequence_dataset_path():
    return RESOURCES_DIR / "data" / "test_image_sequence_dataset"


@pytest.fixture
def video_dataset_path():
    return RESOURCES_DIR / "data" / "test_video_dataset"