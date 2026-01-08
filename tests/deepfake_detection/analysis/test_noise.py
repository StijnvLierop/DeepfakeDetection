import numpy as np
import pytest
from deepfake_detection.analysis.noise import (
    noise_residual,
    channel_noise_imbalance_ratio,
)


@pytest.fixture
def mock_2d_image():
    np.random.seed(42)
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)


@pytest.fixture
def mock_3d_image():
    np.random.seed(42)
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


def test_noise_residual_median_2d_image(mock_2d_image):
    result = noise_residual(mock_2d_image, image_filter="median")
    assert result.shape == mock_2d_image.shape


def test_noise_residual_median_2d_different_from_3d(mock_2d_image, mock_3d_image):
    result = noise_residual(mock_2d_image, image_filter="median")
    result2 = noise_residual(mock_3d_image, image_filter="median")
    assert result.shape != result2.shape


def test_noise_residual_laplace_2d_different_from_3d(mock_2d_image, mock_3d_image):
    result = noise_residual(mock_2d_image, image_filter="laplace")
    result2 = noise_residual(mock_3d_image, image_filter="laplace")
    assert result.shape != result2.shape


def test_noise_residual_3d_image(mock_3d_image):
    result = noise_residual(mock_3d_image, image_filter="median")
    assert result.shape == mock_3d_image.shape


def test_noise_residual_laplace_2d_image(mock_2d_image):
    result = noise_residual(mock_2d_image, image_filter="laplace")
    assert result.shape == mock_2d_image.shape


def test_noise_residual_invalid_filter_raises_error(mock_2d_image):
    with pytest.raises(ValueError):
        noise_residual(mock_2d_image, image_filter="invalid")


def test_channel_noise_imbalance_ratio_returns_float(mock_3d_image):
    result = channel_noise_imbalance_ratio(mock_3d_image)
    assert pytest.approx(result, rel=1e-4) == 0.0058257
    assert isinstance(result, float)


def test_channel_noise_imbalance_ratio_invalid_image_shape_raises_error(mock_2d_image):
    with pytest.raises(ValueError):
        channel_noise_imbalance_ratio(mock_2d_image)
