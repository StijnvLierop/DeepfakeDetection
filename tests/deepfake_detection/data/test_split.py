import numpy as np
import pytest

from deepfake_detection.data import Instance, Dataset
from deepfake_detection.data.datasets import ListDataset
from deepfake_detection.data import split_dataset


@pytest.fixture
def dataset() -> Dataset:
    return ListDataset(instances=[Instance(str(idx), labels) for idx, labels in zip(range(8), (
        {"A"},
        {"A"},
        {"A", "B", "C"},
        {"A", "C"},
        {"B", "C"},
        {"B", "C"},
        {"B", "C"},
        {"C"}
    ))])

def test_split_dataset_size(dataset: Dataset):
    train_set, test_set = split_dataset(dataset, test_size=0.2, random_state=42)
    assert len(train_set) == 6
    assert len(test_set) == 2
    assert len(train_set) + len(test_set) == len(dataset)

    train_set, test_set = split_dataset(dataset, test_size=0.5, random_state=42)
    assert len(train_set) == 4
    assert len(test_set) == 4
    assert len(train_set) + len(test_set) == len(dataset)

def test_split_dataset_random_state(dataset: Dataset):
    train_set1, test_set1 = split_dataset(dataset, test_size=0.2, random_state=42)
    train_set2, test_set2 = split_dataset(dataset, test_size=0.2, random_state=42)
    assert train_set1 == train_set2
    assert test_set1 == test_set2

def test_split_dataset_leakage(dataset: Dataset):
    train_set, test_set = split_dataset(dataset, test_size=0.2, random_state=42)
    for i in train_set:
        assert i not in test_set
    for i in test_set:
        assert i not in train_set

def test_split_dataset_data_unaltered(dataset: Dataset):
    train_set, test_set = split_dataset(dataset, test_size=0.2, random_state=42)
    assert set(train_set.instances + test_set.instances) == set(dataset)