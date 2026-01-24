from deepfake_detection.data.datasets.filter import FilteredDataset


def test_filtered_dataset_iter(dummy_dataset):

    def filter_func(instance):
        return instance.annotation['source'] == "A"

    filtered_dataset = FilteredDataset(dummy_dataset, filter_func)
    filtered_instances = list(filtered_dataset)
    assert len(filtered_instances) == 5


def test_filtered_dataset_empty_result(dummy_dataset):
    filtered_dataset = FilteredDataset(dummy_dataset, lambda x: False)
    assert len(filtered_dataset) == 0
    assert list(filtered_dataset) == []
