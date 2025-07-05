import json
import os
from tempfile import TemporaryDirectory

from deepfake_detection.data.datasets import FileImageDataset
from deepfake_detection.utils.io import load_dataset_from_file
from deepfake_detection.utils.serialization import InstanceEncoder
from tests.deepfake_detection.fixtures import (image_instance, video_instance,
                                               image_sequence_instance, image_dataset_path)


def test_encode_decode_image_instance(image_instance):
    encoded = json.dumps(image_instance, cls=InstanceEncoder)
    decoded = json.loads(encoded)
    assert decoded["path"] == image_instance.path
    assert decoded["label"] == image_instance.label
    assert decoded["instance_type"] == "image"


def test_instance_encoder_with_video_instance(video_instance):
    encoded = json.dumps(video_instance, cls=InstanceEncoder)
    decoded = json.loads(encoded)
    assert decoded["path"] == video_instance.path
    assert decoded["label"] == video_instance.label
    assert decoded["instance_type"] == "video"


def test_instance_encoder_with_image_sequence_instance(image_sequence_instance):
    encoded = json.dumps(image_sequence_instance, cls=InstanceEncoder)
    decoded = json.loads(encoded)
    assert decoded["path"] == image_sequence_instance.path
    assert decoded["label"] == image_sequence_instance.label
    assert decoded["instance_type"] == "image_sequence"


def test_encode_decode_image_dataset(image_dataset_path):
    image_dataset = FileImageDataset(image_dataset_path)
    with TemporaryDirectory() as d:
        tempfile = os.path.join(d, 'test.json')
        image_dataset.save(tempfile)
        loaded_dataset = load_dataset_from_file(tempfile)
        assert set(list(image_dataset)) == set(list(loaded_dataset))