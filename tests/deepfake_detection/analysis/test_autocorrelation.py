import numpy as np
import pytest
from deepfake_detection.analysis.autocorrelation import autocorrelation


def test_autocorrelation_valid_input():
    img = np.array([[1, 2], [3, 4]], dtype=np.float64)
    result = autocorrelation(img)
    assert isinstance(result, np.ndarray)


def test_autocorrelation_output_size():
    img = np.array([[1, 1], [1, 1]], dtype=np.float64)
    result = autocorrelation(img)
    assert result.shape == (3, 3)


def test_autocorrelation_zero_image():
    img = np.zeros((2, 2), dtype=np.float64)
    result = autocorrelation(img)
    expected = np.zeros((3, 3), dtype=np.float64)
    np.testing.assert_array_almost_equal(result, expected)


def test_autocorrelation_non_2d_input():
    img = np.ones((2, 2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="Image must be 2D."):
        autocorrelation(img)


def test_autocorrelation_mean_subtracted():
    img = np.ones((2, 2), dtype=np.float64) * 5
    result = autocorrelation(img)
    expected_autocorr_sum = np.sum(result)
    assert expected_autocorr_sum == pytest.approx(0)
