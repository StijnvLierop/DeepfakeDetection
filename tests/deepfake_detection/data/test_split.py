import collections

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.data.utils import split_dataset

from tests.deepfake_detection.fixtures import dummy_dataset


def test_split_dataset_size(dummy_dataset: Dataset):
    train_set, test_set = split_dataset(dummy_dataset, test_size=0.2, random_state=42)
    assert len(train_set) == 12
    assert len(test_set) == 3
    assert len(train_set) + len(test_set) == len(dummy_dataset)

    train_set, test_set = split_dataset(dummy_dataset, test_size=0.5, random_state=42)
    assert len(train_set) == 7
    assert len(test_set) == 8
    assert len(train_set) + len(test_set) == len(dummy_dataset)


def test_split_dataset_random_state(dummy_dataset: Dataset):
    train_set1, test_set1 = split_dataset(dummy_dataset, test_size=0.2, random_state=42)
    train_set2, test_set2 = split_dataset(dummy_dataset, test_size=0.2, random_state=42)
    assert train_set1 == train_set2
    assert test_set1 == test_set2


def test_split_dataset_leakage(dummy_dataset: Dataset):
    train_set, test_set = split_dataset(dummy_dataset, test_size=0.2, random_state=42)
    overlap = [i for i in train_set if i in test_set]
    assert len(overlap) == 0


def test_split_dataset_data_unaltered(dummy_dataset: Dataset):
    train_set, test_set = split_dataset(dummy_dataset, test_size=0.2, random_state=42)
    assert set(train_set.instances + test_set.instances) == set(dummy_dataset)


def test_split_dataset_stratified_authenticity_label(dummy_dataset: Dataset):
    train_set, test_set = split_dataset(
        dummy_dataset,
        test_size=0.2,
        random_state=42,
        stratify=True,
        label_type="authenticity_label",
    )
    counter_train = collections.Counter(
        [instance.annotation.authenticity_label for instance in train_set]
    )
    counter_test = collections.Counter(
        [instance.annotation.authenticity_label for instance in test_set]
    )
    counter_original = collections.Counter(
        [instance.annotation.authenticity_label for instance in dummy_dataset]
    )
    for label in counter_original:
        train_ratio = counter_train[label] / len(train_set)
        test_ratio = counter_test[label] / len(test_set)
        original_ratio = counter_original[label] / len(dummy_dataset)
        assert abs(train_ratio - original_ratio) < 0.1
        assert abs(test_ratio - original_ratio) < 0.1


def test_split_dataset_stratified_source_label(dummy_dataset: Dataset):
    train_set, test_set = split_dataset(
        dummy_dataset, test_size=0.2, random_state=42, stratify=True
    )
    counter_train = collections.Counter(
        [instance.annotation.source_label for instance in train_set]
    )
    counter_test = collections.Counter(
        [instance.annotation.source_label for instance in test_set]
    )
    counter_original = collections.Counter(
        [instance.annotation.source_label for instance in dummy_dataset]
    )
    for label in counter_original:
        train_ratio = counter_train[label] / len(train_set)
        test_ratio = counter_test[label] / len(test_set)
        original_ratio = counter_original[label] / len(dummy_dataset)
        assert abs(train_ratio - original_ratio) < 0.1
        assert abs(test_ratio - original_ratio) < 0.1
