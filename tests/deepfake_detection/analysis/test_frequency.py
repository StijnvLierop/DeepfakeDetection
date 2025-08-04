import numpy as np
import pytest
from deepfake_detection.analysis.frequency import fft


@pytest.fixture
def sample_image():
    return np.array([[[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                     [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                     [[10, 20, 30], [40, 50, 60], [70, 80, 90]]])


def test_fft_handles_2d_image(sample_image):
    result = fft(sample_image[0])
    assert result.shape == sample_image[0].shape


def test_fft_handles_3d_image(sample_image):
    result = fft(sample_image)
    assert result.shape == sample_image.shape


def test_fft_zero_image():
    zero_image = np.zeros((5, 5))
    result = fft(zero_image)
    assert np.all(result == -np.inf)  # FFT magnitude on zeros should result in -inf (log(0) behavior).
