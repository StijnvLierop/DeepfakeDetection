import os
import pytest


def test_save_raises_error_with_invalid_path_image(image_instance):
    with pytest.raises(ValueError):
        image_instance.save(None)


def test_save_raises_error_with_invalid_path_file_image(file_image_instance):
    with pytest.raises(ValueError):
        file_image_instance.save(None)


def test_save_raises_error_with_invalid_path_file_image_sequence(
    file_image_sequence_instance,
):
    with pytest.raises(ValueError):
        file_image_sequence_instance.save(None)


def test_save_creates_file(image_instance, tmp_path):
    save_path = tmp_path / "output.jpg"
    image_instance.save(save_path)
    assert save_path.exists(), "The file should exist after saving."


def test_save_creates_file_no_file_extension(image_instance, tmp_path):
    save_path = tmp_path / "output"
    save_path = image_instance.save(save_path)
    assert save_path.exists(), "The file should exist after saving."


def test_file_image_instance_save(file_image_instance, tmp_path):
    save_path = tmp_path / "saved_image.jpg"
    file_image_instance.save(save_path)
    assert save_path.exists(), "The file should exist after saving."


def test_file_image_sequence_instance_save(file_image_sequence_instance, tmp_path):
    save_path = tmp_path / "sequence_save"
    file_image_sequence_instance.save(save_path)
    for img in file_image_sequence_instance.data:
        assert os.path.exists(os.path.join(save_path, img.path.name)), (
            "The file should exist after saving."
        )
