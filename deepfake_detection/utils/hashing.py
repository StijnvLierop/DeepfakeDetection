import hashlib

from PIL import Image


def hash_image_to_int(image: Image.Image) -> int:
    return int(hashlib.md5(image.tobytes()).hexdigest(), 16)
