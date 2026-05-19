import pytest

from deepfake_detection.data.datasets import CombinedDataset
from deepfake_detection.data.dataset import Dataset, MapStyleDatasetMixin


# ── helpers ───────────────────────────────────────────────────────────────────


class MockDataset(MapStyleDatasetMixin, Dataset):
    """Map-style dataset backed by a list."""

    def __init__(self, size: int, dataset_name: str):
        super().__init__(dataset_name=dataset_name)
        self.size = size
        self.instances = [f"instance_{i}" for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.instances[idx]


class ItemDataset(MapStyleDatasetMixin, Dataset):
    """Map-style dataset backed by an explicit item list."""

    def __init__(self, items):
        super().__init__(dataset_name="items")
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class StreamingDataset(Dataset):
    """Iterable-only dataset (no __len__, no __getitem__)."""

    def __init__(self, items):
        super().__init__(dataset_name="stream")
        self.items = list(items)

    def __iter__(self):
        yield from self.items


# ── sequential – existing tests ───────────────────────────────────────────────


def test_combined_dataset_length():
    dataset1 = MockDataset(size=5, dataset_name="dataset1")
    dataset2 = MockDataset(size=10, dataset_name="dataset2")
    combined = CombinedDataset([dataset1, dataset2])
    assert len(combined) == 15


def test_combined_dataset_iteration():
    dataset1 = MockDataset(size=3, dataset_name="dataset1")
    dataset2 = MockDataset(size=2, dataset_name="dataset2")
    combined = CombinedDataset([dataset1, dataset2])
    assert list(combined) == [
        "instance_0",
        "instance_1",
        "instance_2",
        "instance_0",
        "instance_1",
    ]


def test_combined_dataset_empty():
    combined = CombinedDataset([])
    assert len(combined) == 0
    assert list(combined) == []


# ── sequential – map-style indexing ──────────────────────────────────────────


def test_combined_sequential_getitem():
    d1 = ItemDataset([10, 20, 30])
    d2 = ItemDataset([40, 50])
    c = CombinedDataset([d1, d2])
    assert c[0] == 10
    assert c[2] == 30  # last of d1
    assert c[3] == 40  # first of d2
    assert c[4] == 50


def test_combined_sequential_negative_index():
    d1 = ItemDataset([1, 2])
    d2 = ItemDataset([3, 4])
    c = CombinedDataset([d1, d2])
    assert c[-1] == 4
    assert c[-4] == 1


def test_combined_sequential_index_out_of_range():
    c = CombinedDataset([ItemDataset([1, 2])])
    with pytest.raises(IndexError):
        _ = c[5]


# ── sequential – streaming disables map-style ─────────────────────────────────


def test_combined_with_streaming_has_no_len():
    c = CombinedDataset([ItemDataset([1, 2]), StreamingDataset([3, 4])])
    with pytest.raises(TypeError):
        len(c)


def test_combined_with_streaming_has_no_getitem():
    c = CombinedDataset([ItemDataset([1, 2]), StreamingDataset([3, 4])])
    with pytest.raises(TypeError):
        _ = c[0]


def test_combined_with_streaming_iterates():
    c = CombinedDataset([ItemDataset([1, 2]), StreamingDataset([3, 4])])
    assert list(c) == [1, 2, 3, 4]


# ── round-robin interleave ────────────────────────────────────────────────────


def test_combined_interleave_equal_sizes():
    d1 = ItemDataset(["a", "b", "c"])
    d2 = ItemDataset([1, 2, 3])
    c = CombinedDataset([d1, d2], interleave=True)
    assert list(c) == ["a", 1, "b", 2, "c", 3]


def test_combined_interleave_unequal_sizes():
    d1 = ItemDataset(["a", "b", "c"])
    d2 = ItemDataset([1, 2])
    c = CombinedDataset([d1, d2], interleave=True)
    # d2 exhausts first; d1 finishes alone
    assert list(c) == ["a", 1, "b", 2, "c"]


def test_combined_interleave_with_streaming():
    d1 = ItemDataset(["a", "b"])
    s = StreamingDataset([1, 2])
    c = CombinedDataset([d1, s], interleave=True)
    assert list(c) == ["a", 1, "b", 2]


def test_combined_interleave_disables_len():
    # interleave=True must disable len() even when all sub-datasets are map-style
    c = CombinedDataset([ItemDataset([1, 2]), ItemDataset([3, 4])], interleave=True)
    with pytest.raises(TypeError):
        len(c)


def test_combined_interleave_disables_getitem():
    c = CombinedDataset([ItemDataset([1, 2])], interleave=True)
    with pytest.raises(TypeError):
        _ = c[0]


def test_combined_interleave_three_datasets():
    d1 = ItemDataset([1])
    d2 = ItemDataset([2])
    d3 = ItemDataset([3])
    c = CombinedDataset([d1, d2, d3], interleave=True)
    assert list(c) == [1, 2, 3]


# ── weighted interleave ───────────────────────────────────────────────────────


def test_combined_weighted_yields_all_items():
    d1 = ItemDataset(list(range(5)))
    d2 = ItemDataset(list("abc"))
    c = CombinedDataset([d1, d2], probabilities=[0.8, 0.2], seed=42)
    result = list(c)
    assert len(result) == 8
    assert set(result) == set(range(5)) | set("abc")


def test_combined_weighted_is_reproducible():
    d1 = ItemDataset(list(range(10)))
    d2 = ItemDataset(list(range(10, 20)))
    result_a = list(CombinedDataset([d1, d2], probabilities=[0.7, 0.3], seed=0))
    result_b = list(CombinedDataset([d1, d2], probabilities=[0.7, 0.3], seed=0))
    assert result_a == result_b


def test_combined_weighted_different_seeds_differ():
    d1 = ItemDataset(list(range(20)))
    d2 = ItemDataset(list(range(20, 40)))
    result_a = list(CombinedDataset([d1, d2], probabilities=[0.5, 0.5], seed=0))
    result_b = list(CombinedDataset([d1, d2], probabilities=[0.5, 0.5], seed=99))
    assert result_a != result_b


def test_combined_weighted_probabilities_mismatch_raises():
    with pytest.raises(ValueError, match="len\\(probabilities\\)"):
        CombinedDataset([ItemDataset([1]), ItemDataset([2])], probabilities=[0.5])


def test_combined_weighted_implies_interleave():
    c = CombinedDataset([ItemDataset([1, 2])], probabilities=[1.0])
    assert c._interleave is True
    with pytest.raises(TypeError):
        len(c)
