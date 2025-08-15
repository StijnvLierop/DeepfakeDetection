from collections import Counter

import pytest

from deepfake_detection.data import sample_n_per_class
from tests.deepfake_detection.fixtures import dummy_dataset


def test_sample_n_per_class(dummy_dataset):
    filtered_dataset = sample_n_per_class(dummy_dataset, n=2)
    assert len(filtered_dataset) == 6
    c = Counter([i.annotation.source_label for i in filtered_dataset])
    for e in c.keys():
        assert c[e] == 2


def test_sample_n_per_class_authenticity(dummy_dataset):
    filtered_dataset = sample_n_per_class(dummy_dataset, n=2, label_type='authenticity_label')
    assert len(filtered_dataset) == 4
    c = Counter([i.annotation.authenticity_label for i in filtered_dataset])
    for e in c.keys():
        assert c[e] == 2


def test_sample_n_per_class_repeatable_when_random_seed(dummy_dataset):
    filtered_dataset_1 = sample_n_per_class(dummy_dataset, n=1, random_seed=42)
    filtered_dataset_2 = sample_n_per_class(dummy_dataset, n=1, random_seed=42)
    assert list(filtered_dataset_1) == list(filtered_dataset_2)


def test_sample_n_per_class_exceed_instances_some_classes(dummy_dataset):
    filtered_dataset = sample_n_per_class(dummy_dataset, n=5)
    assert len(filtered_dataset) == 14


def test_sample_n_per_class_exceed_instances_all_classes(dummy_dataset):
    filtered_dataset = sample_n_per_class(dummy_dataset, n=20)
    assert len(filtered_dataset) == len(dummy_dataset)


def test_sample_n_per_class_invalid_n(dummy_dataset):
    with pytest.raises(ValueError):
        sample_n_per_class(dummy_dataset, n=0)