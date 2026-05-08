from deepfake_detection.data.datasets import CombinedDataset
from deepfake_detection.data.dataset import Dataset, MapStyleDatasetMixin


class MockDataset(MapStyleDatasetMixin, Dataset):
    """A mock dataset implementing the abstract Dataset class for testing."""

    def __init__(self, size: int, dataset_name: str):
        """
        Initialize the mock dataset with a specific size and name.
        :param size: Number of items in the dataset.
        :param dataset_name: Name of the dataset.
        """
        super().__init__(dataset_name=dataset_name)
        self.size = size
        self.instances = [f"instance_{i}" for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.instances[idx]


def test_combined_dataset_length():
    dataset1 = MockDataset(size=5, dataset_name="dataset1")
    dataset2 = MockDataset(size=10, dataset_name="dataset2")
    combined = CombinedDataset([dataset1, dataset2])
    assert len(combined) == 15


def test_combined_dataset_iteration():
    dataset1 = MockDataset(size=3, dataset_name="dataset1")
    dataset2 = MockDataset(size=2, dataset_name="dataset2")
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
