import pytest

from deepfake_detection.data.datasets import ListDataset, MappedDataset
from deepfake_detection.data.instance import Instance
from deepfake_detection.data.annotation import Annotation


class DummyInstance(Instance):
    def __init__(self, value, annotation=None):
        super().__init__(annotation=annotation)
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def save(self, path):
        pass


@pytest.fixture
def dummy_dataset():
    instances = [
        DummyInstance(1, Annotation({"label": "a"})),
        DummyInstance(2, Annotation({"label": "b"})),
    ]
    return ListDataset(dataset_name="test", instances=instances)


def test_map_returns_none(dummy_dataset):
    def mapping_func(instance):
        instance.annotation.set_label("label", "c")

    mapped_ds = MappedDataset(dummy_dataset, mapping_func)

    assert mapped_ds[0] is None


def test_map_returns_instance(dummy_dataset):
    def mapping_func(instance):
        instance.annotation.set_label("label", "c")
        return instance

    mapped_ds = MappedDataset(dummy_dataset, mapping_func)

    assert mapped_ds[0] is not None
    assert mapped_ds[0].annotation.get_label("label") == "c"


def test_map_nested(dummy_dataset):
    def mapping_func1(instance):
        instance.annotation.set_label("label2", "c")
        return instance

    def mapping_func2(instance):
        instance.annotation.set_label("label", {"a": "A", "b": "B"})
        return instance

    mapped_ds = MappedDataset(dummy_dataset, mapping_func1)
    mapped_ds = MappedDataset(mapped_ds, mapping_func2)

    assert mapped_ds[0].annotation.get_label("label2") == "c"
    assert mapped_ds[0].annotation.get_label("label") == "A"
    assert mapped_ds[1].annotation.get_label("label") == "B"
