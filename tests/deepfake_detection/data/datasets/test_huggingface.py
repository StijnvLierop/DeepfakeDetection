from unittest.mock import patch

import pytest

from deepfake_detection.data.datasets.huggingface import (
    HuggingfaceDataset,
    _build_hf_filter,
)
from deepfake_detection.data.instance import ImageInstance


# ── mock HF types ─────────────────────────────────────────────────────────────
#
# We patch two names in the huggingface module:
#   - IterableDataset  → MockIterableDataset
#   - Image            → MockImageFeature
#
# This means the isinstance checks in HuggingfaceDataset.__init__ use our mocks,
# so we can control whether a dataset is seen as streaming or map-style without
# depending on the real HuggingFace classes.


class MockImageFeature:
    """Stand-in for datasets.features.Image."""


class MockIterableDataset:
    """Stand-in for datasets.IterableDataset (streaming mode)."""

    def __init__(self, rows, features=None):
        self.rows = list(rows)
        self.features = features or {"image": MockImageFeature()}

    def __iter__(self):
        yield from self.rows

    def filter(self, func):
        return MockIterableDataset([r for r in self.rows if func(r)], self.features)

    def map(self, func):
        return MockIterableDataset([func(r) for r in self.rows], self.features)

    def shuffle(self, seed=42, buffer_size=1000):
        return MockIterableDataset(self.rows, self.features)

    def take(self, n):
        return MockIterableDataset(self.rows[:n], self.features)


class MockMapDataset:
    """Stand-in for datasets.Dataset (map-style mode)."""

    def __init__(self, rows, features=None):
        self.rows = list(rows)
        self.features = features or {"image": MockImageFeature()}

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        yield from self.rows

    def __getitem__(self, idx):
        return self.rows[idx]

    def filter(self, func):
        return MockMapDataset([r for r in self.rows if func(r)], self.features)

    def map(self, func):
        return MockMapDataset([func(r) for r in self.rows], self.features)

    def shuffle(self, seed=42):
        return MockMapDataset(self.rows, self.features)

    def select(self, indices):
        return MockMapDataset([self.rows[i] for i in indices], self.features)


_ROWS = [
    {"image": None, "label": "real"},
    {"image": None, "label": "fake"},
    {"image": None, "label": "real"},
]

_PATCH_ITERABLE = "deepfake_detection.data.datasets.huggingface.IterableDataset"
_PATCH_IMAGE = "deepfake_detection.data.datasets.huggingface.Image"


# ── _build_hf_filter ──────────────────────────────────────────────────────────


def test_hf_filter_eq_match():
    f = _build_hf_filter({"label": "label", "op": "==", "value": "real"})
    assert f({"label": "real"}) is True


def test_hf_filter_eq_no_match():
    f = _build_hf_filter({"label": "label", "op": "==", "value": "real"})
    assert f({"label": "fake"}) is False


def test_hf_filter_ne():
    f = _build_hf_filter({"label": "label", "op": "!=", "value": "real"})
    assert f({"label": "fake"}) is True
    assert f({"label": "real"}) is False


def test_hf_filter_in():
    f = _build_hf_filter({"label": "label", "op": "in", "value": ["real", "unknown"]})
    assert f({"label": "real"}) is True
    assert f({"label": "fake"}) is False


def test_hf_filter_and():
    f = _build_hf_filter(
        {
            "and": [
                {"label": "label", "op": "==", "value": "real"},
                {"label": "score", "op": ">", "value": 0.5},
            ]
        }
    )
    assert f({"label": "real", "score": 0.8}) is True
    assert f({"label": "real", "score": 0.2}) is False
    assert f({"label": "fake", "score": 0.9}) is False


def test_hf_filter_or():
    f = _build_hf_filter(
        {
            "or": [
                {"label": "label", "op": "==", "value": "real"},
                {"label": "label", "op": "==", "value": "unknown"},
            ]
        }
    )
    assert f({"label": "real"}) is True
    assert f({"label": "unknown"}) is True
    assert f({"label": "fake"}) is False


def test_hf_filter_not():
    f = _build_hf_filter({"not": {"label": "label", "op": "==", "value": "real"}})
    assert f({"label": "real"}) is False
    assert f({"label": "fake"}) is True


def test_hf_filter_nested():
    # (label == real AND score > 0.5) OR label == unknown
    f = _build_hf_filter(
        {
            "or": [
                {
                    "and": [
                        {"label": "label", "op": "==", "value": "real"},
                        {"label": "score", "op": ">", "value": 0.5},
                    ]
                },
                {"label": "label", "op": "==", "value": "unknown"},
            ]
        }
    )
    assert f({"label": "real", "score": 0.9}) is True
    assert f({"label": "real", "score": 0.1}) is False
    assert f({"label": "unknown", "score": 0.0}) is True
    assert f({"label": "fake", "score": 0.9}) is False


def test_hf_filter_missing_key_returns_false():
    # A missing column compares as None, so "==" comparisons typically fail
    f = _build_hf_filter({"label": "nonexistent", "op": "==", "value": "real"})
    assert f({"label": "real"}) is False  # "nonexistent" not in row → None != "real"


def test_hf_filter_inline_func_shorthand():
    # {dotted.func.path: {params}} inline shorthand — no "func" key needed
    f = _build_hf_filter(
        {
            "deepfake_detection.data.utils.filter.hf_hash_filter": {
                "column": "filename",
                "range_min": 0.0,
                "range_max": 1.0,
            }
        }
    )
    assert f({"filename": "img.jpg"}) is True


def test_hf_filter_inline_func_in_compound():
    # inline func shorthand nested inside an "and" block
    f = _build_hf_filter(
        {
            "and": [
                {"label": "label", "op": "==", "value": "real"},
                {
                    "deepfake_detection.data.utils.filter.hf_hash_filter": {
                        "column": "filename",
                        "range_min": 0.0,
                        "range_max": 1.0,
                    }
                },
            ]
        }
    )
    assert f({"label": "real", "filename": "img.jpg"}) is True
    assert f({"label": "fake", "filename": "img.jpg"}) is False


# ── HuggingfaceDataset – map-style ────────────────────────────────────────────


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_hf_nonstreaming_is_not_streaming():
    ds = HuggingfaceDataset(dataset=MockMapDataset(_ROWS), instance_col="image")
    assert ds._streaming is False


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_hf_nonstreaming_len():
    ds = HuggingfaceDataset(dataset=MockMapDataset(_ROWS), instance_col="image")
    assert len(ds) == 3


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_hf_nonstreaming_getitem():
    ds = HuggingfaceDataset(
        dataset=MockMapDataset(_ROWS),
        instance_col="image",
        label_cols={"label": "authenticity"},
    )
    inst = ds[0]
    assert isinstance(inst, ImageInstance)
    assert inst.annotation.get_label("authenticity") == "real"


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_hf_nonstreaming_iter():
    ds = HuggingfaceDataset(
        dataset=MockMapDataset(_ROWS),
        instance_col="image",
        label_cols={"label": "authenticity"},
    )
    labels = [i.annotation.get_label("authenticity") for i in ds]
    assert labels == ["real", "fake", "real"]


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_hf_nonstreaming_no_label_cols():
    ds = HuggingfaceDataset(dataset=MockMapDataset(_ROWS), instance_col="image")
    inst = ds[0]
    assert inst.annotation is None


# ── HuggingfaceDataset – streaming ────────────────────────────────────────────


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_hf_streaming_is_streaming():
    ds = HuggingfaceDataset(dataset=MockIterableDataset(_ROWS), instance_col="image")
    assert ds._streaming is True


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_hf_streaming_len_raises():
    ds = HuggingfaceDataset(dataset=MockIterableDataset(_ROWS), instance_col="image")
    with pytest.raises(TypeError):
        len(ds)


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_hf_streaming_getitem_raises():
    ds = HuggingfaceDataset(dataset=MockIterableDataset(_ROWS), instance_col="image")
    with pytest.raises(TypeError):
        _ = ds[0]


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_hf_streaming_iter():
    ds = HuggingfaceDataset(
        dataset=MockIterableDataset(_ROWS),
        instance_col="image",
        label_cols={"label": "authenticity"},
    )
    assert len(list(ds)) == 3


# ── hf_filter ─────────────────────────────────────────────────────────────────


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_hf_filter_on_map_dataset():
    ds = HuggingfaceDataset(
        dataset=MockMapDataset(_ROWS),
        instance_col="image",
        label_cols={"label": "authenticity"},
        hf_filter={"label": "label", "op": "==", "value": "real"},
    )
    assert len(ds) == 2
    assert all(i.annotation.get_label("authenticity") == "real" for i in ds)


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_hf_filter_on_streaming_dataset():
    ds = HuggingfaceDataset(
        dataset=MockIterableDataset(_ROWS),
        instance_col="image",
        label_cols={"label": "authenticity"},
        hf_filter={"label": "label", "op": "==", "value": "fake"},
    )
    instances = list(ds)
    assert len(instances) == 1
    assert instances[0].annotation.get_label("authenticity") == "fake"


# ── hf_map ────────────────────────────────────────────────────────────────────


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_hf_map_applied():
    ds = HuggingfaceDataset(
        dataset=MockMapDataset(_ROWS),
        instance_col="image",
        label_cols={"label": "authenticity"},
        hf_map=[
            {
                "func": "deepfake_detection.data.utils.map.hf_map_label_values",
                "params": {"label": "label", "value": "normalized"},
            }
        ],
    )
    assert all(i.annotation.get_label("authenticity") == "normalized" for i in ds)


# ── take ──────────────────────────────────────────────────────────────────────


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_take_on_map_dataset():
    ds = HuggingfaceDataset(dataset=MockMapDataset(_ROWS), instance_col="image", take=2)
    assert len(ds) == 2


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_take_on_streaming_dataset():
    ds = HuggingfaceDataset(
        dataset=MockIterableDataset(_ROWS), instance_col="image", take=1
    )
    assert len(list(ds)) == 1


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_take_larger_than_dataset():
    ds = HuggingfaceDataset(
        dataset=MockMapDataset(_ROWS), instance_col="image", take=100
    )
    assert len(ds) == len(_ROWS)


# ── in_memory ─────────────────────────────────────────────────────────────────


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_in_memory_materializes_streaming(monkeypatch):
    materialized = MockMapDataset(_ROWS)

    import deepfake_detection.data.datasets.huggingface as hf_module

    class _FakeDatasets:
        class Dataset:
            @staticmethod
            def from_list(data):
                return materialized

    monkeypatch.setattr(hf_module, "datasets", _FakeDatasets)

    ds = HuggingfaceDataset(
        dataset=MockIterableDataset(_ROWS), instance_col="image", in_memory=True
    )
    assert ds._streaming is False
    assert len(ds) == 3


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_in_memory_no_effect_on_map_dataset():
    ds = HuggingfaceDataset(
        dataset=MockMapDataset(_ROWS), instance_col="image", in_memory=True
    )
    assert ds._streaming is False
    assert len(ds) == 3


# ── unsupported column type raises ────────────────────────────────────────────


@patch(_PATCH_IMAGE, MockImageFeature)
@patch(_PATCH_ITERABLE, MockIterableDataset)
def test_unsupported_column_type_raises():
    class StringFeature:
        pass

    bad_dataset = MockMapDataset(_ROWS, features={"image": StringFeature()})
    with pytest.raises(ValueError, match="not supported"):
        HuggingfaceDataset(dataset=bad_dataset, instance_col="image")
