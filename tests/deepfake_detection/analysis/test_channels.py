import numpy as np
import pytest
from deepfake_detection.analysis.channels import channel_threshold_map


@pytest.fixture
def img():
    # Deterministic 4x4 RGB image with known values
    data = np.zeros((4, 4, 3), dtype=np.uint8)
    data[0, 0] = [200, 50, 50]  # red bright
    data[1, 1] = [50, 200, 50]  # green bright
    data[2, 2] = [50, 50, 200]  # blue bright
    data[3, 3] = [200, 200, 200]  # all channels bright
    return data


def test_output_shape_matches_input(img):
    result = channel_threshold_map(img, channels=[0], threshold=100)
    assert result.shape == img.shape


def test_output_dtype_matches_input(img):
    result = channel_threshold_map(img, channels=[0], threshold=100)
    assert result.dtype == img.dtype


def test_single_channel_any_keeps_matching_pixels(img):
    result = channel_threshold_map(img, channels=[0], threshold=100)
    # (0,0) has red=200 -> passes; (1,1) has red=50 -> zeroed; (3,3) has red=200 -> passes
    assert (result[0, 0] == img[0, 0]).all()
    assert (result[1, 1] == 0).all()
    assert (result[3, 3] == img[3, 3]).all()


def test_mode_any_pixel_passes_if_one_channel_qualifies(img):
    # Channel 0 and 1: (1,1) has green=200 but red=50; should pass with mode='any'
    result = channel_threshold_map(img, channels=[0, 1], threshold=100, mode="any")
    assert (result[1, 1] == img[1, 1]).all()


def test_mode_all_pixel_zeroed_if_not_all_channels_qualify(img):
    # (1,1): red=50, green=200 -> only green qualifies, so 'all' fails
    result = channel_threshold_map(img, channels=[0, 1], threshold=100, mode="all")
    assert (result[1, 1] == 0).all()


def test_mode_all_pixel_passes_when_all_channels_qualify(img):
    # (3,3): red=200, green=200 -> both qualify
    result = channel_threshold_map(img, channels=[0, 1], threshold=100, mode="all")
    assert (result[3, 3] == img[3, 3]).all()


def test_high_threshold_zeroes_all_pixels(img):
    result = channel_threshold_map(img, channels=[0], threshold=255)
    assert (result == 0).all()


def test_zero_threshold_keeps_all_nonzero_pixels(img):
    # With threshold=0, all pixels with any red > 0 pass
    result = channel_threshold_map(img, channels=[0], threshold=0)
    # Every pixel has red >= 50 except the zeroed background pixels
    for r in range(4):
        for c in range(4):
            if img[r, c, 0] > 0:
                assert (result[r, c] == img[r, c]).all()


def test_does_not_modify_input(img):
    original = img.copy()
    channel_threshold_map(img, channels=[0], threshold=100)
    assert (img == original).all()


def test_invalid_mode_raises_value_error(img):
    with pytest.raises(ValueError, match="mode"):
        channel_threshold_map(img, channels=[0], threshold=100, mode="invalid")


def test_invalid_ndim_raises_value_error():
    grayscale = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="ndim"):
        channel_threshold_map(grayscale, channels=[0], threshold=100)


def test_empty_channels_raises_value_error(img):
    with pytest.raises(ValueError, match="channels"):
        channel_threshold_map(img, channels=[], threshold=100)
