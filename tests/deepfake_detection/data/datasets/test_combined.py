from deepfake_detection.data.datasets import CombinedDataset
from deepfake_detection.data.dataset import Dataset


class MockDataset(Dataset):
    """A mock dataset implementing the abstract Dataset class for testing."""

    def __init__(self, size: int, name: str):
        """
        Initialize the mock dataset with a specific size and name.
        :param size: Number of items in the dataset.
        :param name: Name of the dataset.
        """
        super().__init__(name=name)
        self.size = size
        self.instances = [f"instance_{i}" for i in range(size)]

    def __iter__(self):
        return iter(self.instances)

    def __len__(self):
        return self.size


def test_combined_dataset_name():
    dataset1 = MockDataset(size=5, name="dataset1")
    dataset2 = MockDataset(size=10, name="dataset2")
    combined = CombinedDataset([dataset1, dataset2])
    expected_name = "combined_dataset1_dataset2"
    assert combined.name == expected_name


def test_combined_dataset_length():
    dataset1 = MockDataset(size=5, name="dataset1")
    dataset2 = MockDataset(size=10, name="dataset2")
    combined = CombinedDataset([dataset1, dataset2])
    assert len(combined) == 15


def test_combined_dataset_iteration():
    dataset1 = MockDataset(size=3, name="dataset1")
    dataset2 = MockDataset(size=2, name="dataset2")
    combined = CombinedDataset([dataset1, dataset2])
    combined_items = list(combined)
    expected_items = [
        "instance_0",
        "instance_1",
        "instance_2",
        "instance_0",
        "instance_1",
    ]
    assert combined_items == expected_items


def test_combined_dataset_empty():
    combined = CombinedDataset([])
    assert len(combined) == 0
    assert list(combined) == []