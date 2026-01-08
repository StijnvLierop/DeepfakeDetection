import json

from deepfake_detection.utils.serialization import InstanceDecoder, InstanceEncoder


def test_encode_decode_file_image_instance(file_image_instance):
    encoded = json.dumps(file_image_instance, cls=InstanceEncoder)
    decoded = json.loads(encoded, cls=InstanceDecoder)
    assert encoded == decoded


def test_encode_decode_image_instance(image_instance):
    encoded = json.dumps(image_instance, cls=InstanceEncoder)
    decoded = json.loads(encoded, cls=InstanceDecoder)
    assert encoded == decoded


def test_instance_encoder_with_video_instance(file_video_instance):
    encoded = json.dumps(file_video_instance, cls=InstanceEncoder)
    decoded = json.loads(encoded, cls=InstanceDecoder)
    assert encoded == decoded


def test_instance_encoder_with_image_sequence_instance(file_image_sequence_instance):
    encoded = json.dumps(file_image_sequence_instance, cls=InstanceEncoder)
    decoded = json.loads(encoded, cls=InstanceDecoder)
    assert encoded == decoded
