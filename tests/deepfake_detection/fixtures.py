import os

import pytest

from deepfake_detection.data import ImageInstance, VideoInstance, ImageSequenceInstance
from tests.deepfake_detection.paths import RESOURCES_DIR


@pytest.fixture
def image_instance():
    return ImageInstance(path=os.path.join(RESOURCES_DIR, "/data/test_image_dataset/fake/model1/fake01.png"),
                         label="fake")


@pytest.fixture
def video_instance():
    return VideoInstance(path=os.path.join(RESOURCES_DIR, "/data/test_image_dataset/fake/model1/fake1.mp4"),
                         label="fake")


@pytest.fixture
def image_sequence_instance():
    return ImageSequenceInstance(path=os.path.join(RESOURCES_DIR, "/data/test_image_sequence_dataset/fake/model1/0_0fake"),
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