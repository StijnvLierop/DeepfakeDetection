import json
import os
from tempfile import TemporaryDirectory

from deepfake_detection.data.datasets import FileImageDataset
from deepfake_detection.utils.io import load_dataset_from_file
from deepfake_detection.utils.serialization import (FileImageInstanceEncoder, FileVideoInstanceEncoder,
                                                    FileImageSequenceInstanceEncoder, ImageInstanceEncoder,
                                                    FileImageInstanceDecoder, ImageInstanceDecoder,
                                                    FileVideoInstanceDecoder, FileImageSequenceInstanceDecoder)
from tests.deepfake_detection.fixtures import (image_instance, file_image_instance, file_video_instance,
                                               file_image_sequence_instance, image_dataset_path)


def test_encode_decode_file_image_instance(file_image_instance):
    encoded = json.dumps(file_image_instance, cls=FileImageInstanceEncoder)
    decoded = json.loads(encoded, cls=FileImageInstanceDecoder)
    assert encoded == decoded

def test_encode_decode_image_instance(image_instance):
    encoded = json.dumps(image_instance, cls=ImageInstanceEncoder)
    decoded = json.loads(encoded, cls=ImageInstanceDecoder)
    assert encoded == decoded

def test_instance_encoder_with_video_instance(file_video_instance):
    encoded = json.dumps(file_video_instance, cls=FileVideoInstanceEncoder)
    decoded = json.loads(encoded, cls=FileVideoInstanceDecoder)
    assert encoded == decoded

def test_instance_encoder_with_image_sequence_instance(file_image_sequence_instance):
    encoded = json.dumps(file_image_sequence_instance, cls=FileImageSequenceInstanceEncoder)
    decoded = json.loads(encoded, cls=FileImageSequenceInstanceDecoder)
    assert encoded == decoded

def test_encode_decode_image_dataset(image_dataset_path):
    image_dataset = FileImageDataset(image_dataset_path)
    with TemporaryDirectory() as d:
        tempfile = os.path.join(d, 'test.json')
        image_dataset.save(tempfile, encoder=FileImageInstanceEncoder)
        loaded_dataset = load_dataset_from_file(tempfile, decoder=FileImageInstanceDecoder)
        assert set(list(image_dataset)) == set(list(loaded_dataset))