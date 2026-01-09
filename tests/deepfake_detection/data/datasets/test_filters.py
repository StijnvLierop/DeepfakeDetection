import functools
from collections import Counter

import pytest

from data.datasets import FilteredDataset
from deepfake_detection.data.filters import sample_n_per_class_filter


def test_sample_n_per_class(dummy_dataset):
    filter_func = functools.partial(sample_n_per_class_filter, n=2)
    filtered_dataset = FilteredDataset(dummy_dataset, filter_func)
    assert len(filtered_dataset) == 6
    c = Counter([i.annotation.source_label for i in filtered_dataset])
    for e in c.keys():
        assert c[e] == 2


def test_sample_n_per_class_authenticity(dummy_dataset):
    filter_func = functools.partial(sample_n_per_class_filter, n=2, label_type="authenticity_label")
    filtered_dataset = FilteredDataset(dummy_dataset, filter_func)
    assert len(filtered_dataset) == 4
    c = Counter([i.annotation.authenticity_label for i in filtered_dataset])
    for e in c.keys():
        assert c[e] == 2


def test_sample_n_per_class_repeatable_when_random_seed(dummy_dataset):
    filter_func = functools.partial(sample_n_per_class_filter, n=1, random_seed=42)
    filtered_dataset_1 = FilteredDataset(dummy_dataset, filter_func)
    filtered_dataset_2 = FilteredDataset(dummy_dataset, filter_func)
    assert list(filtered_dataset_1) == list(filtered_dataset_2)


def test_sample_n_per_class_exceed_instances_some_classes(dummy_dataset):
    filter_func = functools.partial(sample_n_per_class_filter, n=5)
    filtered_dataset = FilteredDataset(dummy_dataset, filter_func)
    assert len(filtered_dataset) == 14


def test_sample_n_per_class_exceed_instances_all_classes(dummy_dataset):
    filter_func = functools.partial(sample_n_per_class_filter, n=20)
    filtered_dataset = FilteredDataset(dummy_dataset, filter_func)
    assert len(filtered_dataset) == len(dummy_dataset)


def test_sample_n_per_class_invalid_n(dummy_dataset):
    with pytest.raises(ValueError):
        filter_func = functools.partial(sample_n_per_class_filter, n=0)
        FilteredDataset(dummy_dataset, filter_func)
