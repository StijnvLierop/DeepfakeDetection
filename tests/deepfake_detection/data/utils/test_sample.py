from collections import Counter

import pytest

from deepfake_detection.data.utils.sample import sample_n_per_class


def test_sample_n_per_class(dummy_dataset):
    dataset = sample_n_per_class(dummy_dataset, n=2, label='source')
    assert len(dataset) == 6
    c = Counter([i.annotation["source"] for i in dataset])
    for e in c.keys():
        assert c[e] == 2


def test_sample_n_per_class_authenticity(dummy_dataset):
    dataset = sample_n_per_class(dummy_dataset, n=2, label="authenticity")
    assert len(dataset) == 4
    c = Counter([i.annotation["authenticity"] for i in dataset])
    for e in c.keys():
        assert c[e] == 2


def test_sample_n_per_class_repeatable_when_random_seed(dummy_dataset):
    dataset_1 = sample_n_per_class(dummy_dataset, n=1, random_state=42, label='source')
    dataset_2 = sample_n_per_class(dummy_dataset, n=1, random_state=42, label='source')
    assert list(dataset_1) == list(dataset_2)


def test_sample_n_per_class_exceed_instances_some_classes(dummy_dataset):
    dataset = sample_n_per_class(dummy_dataset, n=5, label='source')
    assert len(dataset) == 14


def test_sample_n_per_class_exceed_instances_all_classes(dummy_dataset):
    dataset = sample_n_per_class(dummy_dataset, n=20, label='source')
    assert len(dataset) == len(dummy_dataset)


def test_sample_n_per_class_invalid_n(dummy_dataset):
    with pytest.raises(ValueError):
        _ = sample_n_per_class(dummy_dataset, n=0, label='source')