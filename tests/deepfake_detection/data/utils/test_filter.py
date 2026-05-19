import hashlib
import pytest
from unittest.mock import MagicMock

from deepfake_detection.data.utils.filter import filter_on_hash_value, hf_hash_filter
from deepfake_detection.data.instance import Instance


# ── hf_hash_filter ────────────────────────────────────────────────────────────


def _bucket(value: str) -> float:
    """Expected hash bucket for a value — mirrors the implementation."""
    return int(hashlib.md5(value.encode()).hexdigest(), 16) % 100 / 100


def test_hf_hash_filter_full_range_keeps_all():
    assert (
        hf_hash_filter(
            {"filename": "img.jpg"}, column="filename", range_min=0.0, range_max=1.0
        )
        is True
    )


def test_hf_hash_filter_missing_column_returns_false():
    assert hf_hash_filter({"other": "value"}, column="filename") is False


def test_hf_hash_filter_none_value_returns_false():
    assert hf_hash_filter({"filename": None}, column="filename") is False


def test_hf_hash_filter_lower_bound_inclusive():
    b = _bucket("img.jpg")
    assert (
        hf_hash_filter(
            {"filename": "img.jpg"},
            "filename",
            range_min=b,
            range_max=min(b + 0.01, 1.0),
        )
        is True
    )


def test_hf_hash_filter_upper_bound_exclusive():
    b = _bucket("img.jpg")
    assert (
        hf_hash_filter(
            {"filename": "img.jpg"},
            "filename",
            range_min=max(b - 0.01, 0.0),
            range_max=b,
        )
        is False
    )


def test_hf_hash_filter_stable_across_calls():
    row = {"filename": "stable.jpg"}
    r1 = hf_hash_filter(row, "filename", range_min=0.0, range_max=0.5)
    r2 = hf_hash_filter(row, "filename", range_min=0.0, range_max=0.5)
    assert r1 == r2


def test_hf_hash_filter_no_split_overlap():
    """train and val splits are disjoint and together cover all rows."""
    rows = [{"filename": f"img_{i:04d}.jpg"} for i in range(200)]
    train = [r for r in rows if hf_hash_filter(r, "filename", 0.0, 0.8)]
    val = [r for r in rows if hf_hash_filter(r, "filename", 0.8, 1.0)]
    assert set(r["filename"] for r in train) & set(r["filename"] for r in val) == set()
    assert len(train) + len(val) == len(rows)


def test_hf_hash_filter_same_value_same_split():
    """Two rows sharing a column value always land in the same split (no frame leakage)."""
    row1 = {"filename": "video_001_frame_01.jpg", "label": "real"}
    row2 = {"filename": "video_001_frame_01.jpg", "label": "fake"}
    assert hf_hash_filter(row1, "filename", 0.0, 0.8) == hf_hash_filter(
        row2, "filename", 0.0, 0.8
    )


# ── filter_on_hash_value ──────────────────────────────────────────────────────


@pytest.fixture
def mock_instance():
    """Provides a fresh MagicMock of the Instance class."""
    return MagicMock(spec=Instance)


def test_hash_exactly_at_lower_bound(mock_instance):
    """
    If range is 0.2 to 0.5, the lower bound is bucket 20.
    A hash of 120 (120 % 100 = 20) should return True.
    """
    mock_instance.__hash__.return_value = 120

    # range_min=0.2 (20.0) <= bucket=20 < range_max=0.5 (50.0)
    result = filter_on_hash_value(mock_instance, range_min=0.2, range_max=0.5)
    assert result is True


def test_hash_exactly_at_upper_bound(mock_instance):
    """
    If range is 0.2 to 0.5, the upper bound is bucket 50.
    A hash of 50 should return False (exclusive upper bound).
    """
    mock_instance.__hash__.return_value = 50

    result = filter_on_hash_value(mock_instance, range_min=0.2, range_max=0.5)
    assert result is False


def test_negative_hash_handling(mock_instance):
    """
    The function uses abs(), so a hash of -10 should behave like bucket 10.
    """
    mock_instance.__hash__.return_value = -10

    # 0.0 <= 10 < 0.2 (Range is 0 to 20)
    result = filter_on_hash_value(mock_instance, range_min=0.0, range_max=0.2)
    assert result is True


def test_hash_out_of_range(mock_instance):
    """Should return False for values clearly outside the range."""
    # Hash 85 % 100 = 85. Range is 20-50.
    mock_instance.__hash__.return_value = 85

    result = filter_on_hash_value(mock_instance, range_min=0.2, range_max=0.5)
    assert result is False


def test_floating_point_range(mock_instance):
    """
    Check precision for small ranges.
    Range 0.05 to 0.06 is bucket 5.
    """
    mock_instance.__hash__.return_value = 5
    result = filter_on_hash_value(mock_instance, range_min=0.05, range_max=0.06)
    assert result is True


def test_data_leakage_between_ranges():
    """
    Verifies that two adjacent hash ranges result in zero overlapping instances.
    """
    # Create a pool of 1,000 mock instances with unique hashes
    dataset_pool = []
    for i in range(1000):
        mock_inst = MagicMock()
        # Assign a deterministic hash for testing
        mock_inst.__hash__.return_value = i
        dataset_pool.append(mock_inst)

    # Split the pool into two sets using your filter function
    # Range A: 0% to 80% (buckets 0-79)
    # Range B: 80% to 100% (buckets 80-99)
    train_set = [inst for inst in dataset_pool if filter_on_hash_value(inst, 0.0, 0.8)]
    test_set = [inst for inst in dataset_pool if filter_on_hash_value(inst, 0.8, 1.0)]

    # Check for leakage (Intersection)
    # Convert to sets of IDs (or the mocks themselves) to check for overlap
    intersection = set(train_set).intersection(set(test_set))

    # 4. Assertions
    assert len(intersection) == 0, (
        f"Data leakage detected! {len(intersection)} instances in both sets."
    )
    assert len(train_set) + len(test_set) == len(dataset_pool), (
        "Data loss detected in split logic."
    )


def test_identity_leakage_logic():
    """
    Ensures that two different objects with the same hash value
    always end up in the same split.
    """
    # Two different objects that represent the same entity (e.g., same Video ID)
    instance_v1_frame1 = MagicMock()
    instance_v1_frame1.__hash__.return_value = 12345

    instance_v1_frame2 = MagicMock()
    instance_v1_frame2.__hash__.return_value = 12345

    # Check if they both fall into the same range
    in_train_1 = filter_on_hash_value(instance_v1_frame1, 0.0, 0.8)
    in_train_2 = filter_on_hash_value(instance_v1_frame2, 0.0, 0.8)

    assert in_train_1 == in_train_2, (
        "Identity leakage: Same hash resulted in different splits."
    )
