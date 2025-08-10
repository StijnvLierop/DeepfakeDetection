import numpy as np
import pytest
from deepfake_detection.analysis.chromatic_aberration import (
    calc_vector_field,
    correlation_coefficient,
    estimate_lateral_chromatic_aberration,
    diamond_search,
    simulate_lateral_chromatic_aberration, estimate_local_lca_displacements,
)


def test_calc_vector_field_output_shapes():
    img = np.zeros((100, 100), dtype=np.float32)
    x0, y0 = 50, 50
    alpha = 1.2
    step = 10

    x, y, xw, yw = calc_vector_field(img, x0, y0, alpha, step)

    assert x.shape == y.shape
    assert x.shape == xw.shape
    assert x.shape == yw.shape


def test_calc_vector_field_values():
    img = np.zeros((50, 50), dtype=np.float32)
    x0, y0 = 25, 25
    alpha = 1.5
    step = 5

    x, y, xw, yw = calc_vector_field(img, x0, y0, alpha, step)

    assert xw[0, 0] == pytest.approx(alpha * (0 - x0) + x0 - 0)
    assert yw[0, 0] == pytest.approx(alpha * (0 - y0) + y0 - 0)


def test_calc_vector_field_step_size():
    img = np.zeros((60, 60), dtype=np.float32)
    x0, y0 = 30, 30
    alpha = 1.0
    step = 20

    x, y, _, _ = calc_vector_field(img, x0, y0, alpha, step)

    assert x.shape == (3, 3)
    assert y.shape == (3, 3)


def test_correlation_coefficient_basic():
    block1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    block2 = np.array([[2, 4], [6, 8]], dtype=np.float32)
    result = correlation_coefficient(block1, block2)
    assert result == pytest.approx(1.0)


def test_simulate_lateral_chromatic_aberration_output_shape():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    alpha = 1.2
    output = simulate_lateral_chromatic_aberration(img, alpha)
    assert output.shape == img.shape


def test_simulate_lateral_chromatic_aberration_no_change_alpha_1():
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    alpha = 1.0
    output = simulate_lateral_chromatic_aberration(img, alpha)
    assert np.array_equal(output, img)


def test_simulate_lateral_chromatic_aberration_custom_center():
    img = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
    alpha = 1.3
    center = (15, 20)
    output = simulate_lateral_chromatic_aberration(img, alpha, center)
    assert output.shape == img.shape  # Ensure shape remains unchanged


def test_correlation_coefficient_with_zero_denominator():
    block1 = np.zeros((2, 2), dtype=np.float32)
    block2 = np.zeros((2, 2), dtype=np.float32)
    result = correlation_coefficient(block1, block2)
    assert result == 0


def test_correlation_coefficient_negative():
    block1 = np.array([[1, -2], [-3, 4]], dtype=np.float32)
    block2 = np.array([[-1, 2], [3, -4]], dtype=np.float32)
    result = correlation_coefficient(block1, block2)
    assert result == pytest.approx(-1.0)


def test_estimate_lateral_chromatic_aberration_output_shapes():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    local_displacements_gr, local_displacements_gb, global_disp_gr, global_disp_gb, keypoints = (
        estimate_lateral_chromatic_aberration(img)
    )

    assert local_displacements_gr.shape == local_displacements_gb.shape
    assert global_disp_gr.shape == (3,)
    assert global_disp_gb.shape == (3,)
    assert keypoints.shape[1] == 2


def test_estimate_local_lca_displacements_with_single_keypoint():
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    keypoints = np.array([[25, 25]])
    block_size = 3
    search_range = 2

    local_displacements_gr, local_displacements_gb = estimate_local_lca_displacements(img, keypoints, block_size,
                                                                                      search_range)

    assert local_displacements_gr.shape == (1, 2)
    assert local_displacements_gb.shape == (1, 2)


def test_estimate_local_lca_displacements_with_multiple_keypoints():
    img = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
    keypoints = np.array([[15, 15], [30, 30], [45, 45]])
    block_size = 5
    search_range = 3

    local_displacements_gr, local_displacements_gb = estimate_local_lca_displacements(img, keypoints, block_size,
                                                                                      search_range)

    assert local_displacements_gr.shape == (3, 2)
    assert local_displacements_gb.shape == (3, 2)


def test_estimate_local_lca_displacements_with_edge_keypoint():
    img = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
    keypoints = np.array([[0, 0]])  # Keypoint at the top-left corner
    block_size = 5
    search_range = 2

    # Assuming behavior is valid blocks only, these edge cases may result in no correlations
    local_displacements_gr, local_displacements_gb = estimate_local_lca_displacements(img, keypoints, block_size,
                                                                                      search_range)

    assert local_displacements_gr.shape == (1, 2)
    assert local_displacements_gb.shape == (1, 2)


def test_diamond_search_with_perfect_match():
    template = np.array([[1, 2], [3, 4]], dtype=np.float32)
    search_area = np.array([[0, 0, 0, 0],
                            [0, 1, 2, 0],
                            [0, 3, 4, 0],
                            [0, 0, 0, 0]], dtype=np.float32)

    dx, dy = diamond_search(template, search_area)
    assert dx == 0
    assert dy == 0


def test_diamond_search_with_multiple_candidates():
    template = np.array([[1, 1], [1, 1]], dtype=np.float32)
    search_area = np.array([[0, 1, 1, 0],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [0, 1, 1, 0]], dtype=np.float32)

    dx, dy = diamond_search(template, search_area)
    assert (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_diamond_search_with_no_match():
    template = np.array([[1, 2], [3, 4]], dtype=np.float32)
    search_area = np.zeros((4, 4), dtype=np.float32)
    dx, dy = diamond_search(template, search_area)
    assert dx == 0
    assert dy == 0


def test_estimate_lateral_chromatic_aberration_keypoints_existence():
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    _, _, _, _, keypoints = estimate_lateral_chromatic_aberration(img)
    assert len(keypoints) > 0