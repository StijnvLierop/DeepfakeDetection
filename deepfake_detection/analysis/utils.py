import numpy as np


def normalize_img(img: np.ndarray) -> np.ndarray:
    """
    Normalize an image to the range [0, 255].

    :param img: The image to normalize.
    :return: The normalized image.
    """
    return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)