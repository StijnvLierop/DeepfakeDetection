import numpy as np
import pytest
from PIL import Image

from deepfake_detection.analysis.utils import average_over_images
from deepfake_detection.data.instance import ImageInstance


@pytest.fixture
def image_instance():
    data = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8))
    return ImageInstance(data)


@pytest.fixture
def image_instance2():
    data = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 3)
    return ImageInstance(data)


def test_average_over_images(image_instance, image_instance2):
    def dummy_function(x):
        return x + 5

    instances = [image_instance, image_instance2]
    result = average_over_images(instances, dummy_function)

    assert isinstance(result, np.ndarray)
    assert result.shape[:2] == (100, 100)
    assert np.all(result == 7)


def test_average_over_images_no_instances():
    with pytest.raises(IndexError):
        average_over_images([], lambda x: x)
