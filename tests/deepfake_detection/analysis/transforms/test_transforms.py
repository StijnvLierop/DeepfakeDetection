import numpy as np
import pytest

from deepfake_detection.analysis.channels import channel_threshold_map
from deepfake_detection.analysis.frequency import fft
from deepfake_detection.analysis.noise import noise_residual
from deepfake_detection.analysis.prnu import prnu_fstv
from deepfake_detection.analysis.transforms import (
    AutocorrelationTransform,
    ChannelThresholdMap,
    ELATransform,
    FFTTransform,
    NoiseResidualTransform,
    PRNUTransform,
    TransformPipeline,
)


@pytest.fixture
def img_3d():
    np.random.seed(0)
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def img_2d():
    np.random.seed(0)
    return np.random.randint(0, 256, (64, 64), dtype=np.uint8)


# ---------------------------------------------------------------------------
# ChannelThresholdMap
# ---------------------------------------------------------------------------

class TestChannelThresholdMap:
    def test_apply_matches_underlying_function(self, img_3d):
        t = ChannelThresholdMap(channels=[0], threshold=100)
        expected = channel_threshold_map(img_3d, [0], 100, "any")
        np.testing.assert_array_equal(t.apply(img_3d), expected)

    def test_name_encodes_config(self):
        assert ChannelThresholdMap([0, 2], 128, "all").name == "channel_threshold_ch0_2_t128_all_above"

    def test_name_default_mode(self):
        assert ChannelThresholdMap([1], 50).name == "channel_threshold_ch1_t50_any_above"

    def test_apply_does_not_mutate_input(self, img_3d):
        original = img_3d.copy()
        ChannelThresholdMap([0], 100).apply(img_3d)
        np.testing.assert_array_equal(img_3d, original)

    def test_output_shape_matches_input(self, img_3d):
        result = ChannelThresholdMap([0], 100).apply(img_3d)
        assert result.shape == img_3d.shape


# ---------------------------------------------------------------------------
# FFTTransform
# ---------------------------------------------------------------------------

class TestFFTTransform:
    def test_apply_matches_underlying_function(self, img_3d):
        t = FFTTransform()
        expected = fft(img_3d.copy(), hamming_window=False)
        np.testing.assert_array_almost_equal(t.apply(img_3d), expected)

    def test_name_no_hamming(self):
        assert FFTTransform().name == "fft"

    def test_name_hamming(self):
        assert FFTTransform(hamming_window=True).name == "fft_hamming"

    def test_apply_does_not_mutate_input(self, img_3d):
        original = img_3d.copy()
        FFTTransform().apply(img_3d)
        np.testing.assert_array_equal(img_3d, original)


# ---------------------------------------------------------------------------
# NoiseResidualTransform
# ---------------------------------------------------------------------------

class TestNoiseResidualTransform:
    def test_apply_matches_underlying_function(self, img_3d):
        t = NoiseResidualTransform()
        expected = noise_residual(img_3d, image_filter="median")
        np.testing.assert_array_almost_equal(t.apply(img_3d), expected)

    def test_name_median(self):
        assert NoiseResidualTransform("median").name == "noise_residual_median"

    def test_name_laplace(self):
        assert NoiseResidualTransform("laplace").name == "noise_residual_laplace"

    def test_output_shape_matches_input(self, img_3d):
        result = NoiseResidualTransform().apply(img_3d)
        assert result.shape == img_3d.shape


# ---------------------------------------------------------------------------
# PRNUTransform
# ---------------------------------------------------------------------------

class TestPRNUTransform:
    def test_apply_matches_underlying_function(self, img_3d):
        expected = prnu_fstv(img_3d)
        np.testing.assert_array_almost_equal(PRNUTransform().apply(img_3d), expected)

    def test_name(self):
        assert PRNUTransform().name == "prnu"

    def test_output_shape_matches_input(self, img_3d):
        assert PRNUTransform().apply(img_3d).shape == img_3d.shape


# ---------------------------------------------------------------------------
# AutocorrelationTransform
# ---------------------------------------------------------------------------

class TestAutocorrelationTransform:
    def test_apply_3d_returns_2d(self, img_3d):
        result = AutocorrelationTransform().apply(img_3d)
        assert result.ndim == 2

    def test_apply_2d_returns_2d(self, img_2d):
        result = AutocorrelationTransform().apply(img_2d)
        assert result.ndim == 2

    def test_name(self):
        assert AutocorrelationTransform().name == "autocorrelation"

    def test_output_is_larger_than_input(self, img_3d):
        # fftconvolve with mode="full" produces (2*H-1, 2*W-1) output
        result = AutocorrelationTransform().apply(img_3d)
        h, w = img_3d.shape[:2]
        assert result.shape == (2 * h - 1, 2 * w - 1)


# ---------------------------------------------------------------------------
# ELATransform
# ---------------------------------------------------------------------------

class TestELATransform:
    def test_apply_returns_ndarray(self, img_3d):
        result = ELATransform().apply(img_3d)
        assert isinstance(result, np.ndarray)

    def test_apply_output_shape_matches_input(self, img_3d):
        result = ELATransform().apply(img_3d)
        assert result.shape == img_3d.shape

    def test_name(self):
        assert ELATransform().name == "ela_q95"

    def test_name_custom_quality(self):
        assert ELATransform(quality=75).name == "ela_q75"

    def test_apply_does_not_mutate_input(self, img_3d):
        original = img_3d.copy()
        ELATransform().apply(img_3d)
        np.testing.assert_array_equal(img_3d, original)


# ---------------------------------------------------------------------------
# TransformPipeline
# ---------------------------------------------------------------------------

class TestTransformPipeline:
    def test_name_joins_member_names(self):
        pipeline = TransformPipeline(PRNUTransform(), FFTTransform())
        assert pipeline.name == "prnu__fft"

    def test_single_transform_pipeline(self, img_3d):
        t = PRNUTransform()
        pipeline = TransformPipeline(t)
        np.testing.assert_array_almost_equal(pipeline.apply(img_3d), t.apply(img_3d))

    def test_chaining_applies_in_order(self, img_3d):
        pipeline = TransformPipeline(PRNUTransform(), FFTTransform())
        expected = FFTTransform().apply(PRNUTransform().apply(img_3d))
        np.testing.assert_array_almost_equal(pipeline.apply(img_3d), expected)

    def test_empty_pipeline_raises(self):
        with pytest.raises(ValueError):
            TransformPipeline()

    def test_output_shape_after_prnu_fft(self, img_3d):
        result = TransformPipeline(PRNUTransform(), FFTTransform()).apply(img_3d)
        assert result.shape == img_3d.shape
