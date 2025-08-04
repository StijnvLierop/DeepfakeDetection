import numpy as np
import pytest
from deepfake_detection.analysis.frequency import fft


@pytest.fixture
def sample_image():
    """Fixture that returns a sample 2D numpy array as a test image."""
    return np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])


def test_fft_returns_array(sample_image):
    """Test if fft function returns a numpy array."""
    result = fft(sample_image)
    assert isinstance(result, np.ndarray)


def test_fft_normalization(sample_image):
    """Test fft function with normalization enabled."""
    result = fft(sample_image, normalize=True)
    assert result.max() <= 255
    assert result.min() >= 0


def test_fft_no_normalization(sample_image):
    """Test fft function with normalization disabled."""
    result = fft(sample_image, normalize=False)
    assert result.max() > 255
    assert result.min() < 0


def test_fft_zero_image():
    """Test fft function with an input of all zeros."""
    zero_image = np.zeros((5, 5))
    result = fft(zero_image)
    assert np.all(result == -np.inf)  # FFT magnitude on zeros should result in -inf (log(0) behavior).


def test_fft_large_image():
    """Test fft function with a larger image."""
    large_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    result = fft(large_image, normalize=True)
    assert result.shape == large_image.shape
    assert result.max() <= 255
    assert result.min() >= 0
