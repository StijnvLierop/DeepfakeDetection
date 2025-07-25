import collections
import itertools

import numpy as np
import pytest
from PIL import Image

from deepfake_detection.data import Dataset, ImageInstance
from deepfake_detection.data import split_dataset
from deepfake_detection.data.datasets import ListDataset


@pytest.fixture
def dataset() -> Dataset:
    return ListDataset(
        instances=[ImageInstance(data=Image.fromarray(np.zeros(100+d)), label=l) for (d, l) in
                    zip(range(15),
                        ["A", "A", "B", "C", "C", "C", "B", "C", "C", "A", "A", "B", "C", "A", "B"])
                   ]
    )

def test_split_dataset_size(dataset: Dataset):
    train_set, test_set = split_dataset(dataset, test_size=0.2, random_state=42)
    assert len(train_set) == 12
    assert len(test_set) == 3
    assert len(train_set) + len(test_set) == len(dataset)

    train_set, test_set = split_dataset(dataset, test_size=0.5, random_state=42)
    assert len(train_set) == 7
    assert len(test_set) == 8
    assert len(train_set) + len(test_set) == len(dataset)

def test_split_dataset_random_state(dataset: Dataset):
    train_set1, test_set1 = split_dataset(dataset, test_size=0.2, random_state=42)
    train_set2, test_set2 = split_dataset(dataset, test_size=0.2, random_state=42)
    assert train_set1 == train_set2
    assert test_set1 == test_set2

def test_split_dataset_leakage(dataset: Dataset):
    train_set, test_set = split_dataset(dataset, test_size=0.2, random_state=42)
    overlap = [i for i in train_set if i in test_set]
    assert len(overlap) == 0

def test_split_dataset_data_unaltered(dataset: Dataset):
    train_set, test_set = split_dataset(dataset, test_size=0.2, random_state=42)
    assert set(train_set.instances + test_set.instances) == set(dataset)


def test_split_dataset_stratified(dataset: Dataset):
    train_set, test_set = split_dataset(dataset, test_size=0.2, random_state=42, stratify=True)
    counter_train = collections.Counter([instance.label for instance in train_set])
    counter_test = collections.Counter([instance.label for instance in test_set])
    counter_original = collections.Counter([instance.label for instance in dataset])
    for label in counter_original:
        train_ratio = counter_train[label] / len(train_set)
        test_ratio = counter_test[label] / len(test_set)
        original_ratio = counter_original[label] / len(dataset)
        assert abs(train_ratio - original_ratio) < 0.1
        assert abs(test_ratio - original_ratio) < 0.1