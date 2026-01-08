import numpy as np
import pytest
from PIL import Image
from deepfake_detection.analysis.prnu import prnu_fstv
from deepfake_detection.data.instance import ImageInstance, FileImageInstance


@pytest.fixture
def mock_image_instance():
    data = Image.fromarray(np.full((100, 100), 128, dtype=np.uint8))
    return ImageInstance(data)


@pytest.fixture
def mock_image_instances():
    img1 = ImageInstance(
        data=Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
    )
    img2 = ImageInstance(
        data=Image.fromarray(np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8))
    )
    img3 = ImageInstance(
        data=Image.fromarray(np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8))
    )
    return [img1, img2, img3]


@pytest.fixture
def mock_file_image_instances(tmp_path):
    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"
    img3_path = tmp_path / "img3.png"
    Image.fromarray(np.random.randint(0, 256, (120, 120, 3), dtype=np.uint8)).save(
        img1_path
    )
    Image.fromarray(np.random.randint(0, 256, (140, 140, 3), dtype=np.uint8)).save(
        img2_path
    )
    Image.fromarray(np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)).save(
        img3_path
    )
    img1 = FileImageInstance(path=img1_path)
    img2 = FileImageInstance(path=img2_path)
    img3 = FileImageInstance(path=img3_path)
    return [img1, img2, img3]


def test_prnu_fstv_returns_array(mock_image_instance):
    result = prnu_fstv(mock_image_instance)
    assert isinstance(result, np.ndarray)


def test_prnu_fstv_shape_equal_to_image_dimensions(mock_image_instance):
    result = prnu_fstv(mock_image_instance)
    assert result.shape == (100, 100)
