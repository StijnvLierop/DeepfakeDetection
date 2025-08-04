import numpy as np


def normalize_img(img: np.ndarray) -> np.ndarray:
    """
    Normalize an image to the range [0, 255].

    :param img: The image to normalize.
    :return: The normalized image.
    """
    return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)


def centercrop(img: np.ndarray, min_width: int, min_height: int) -> np.ndarray:
    """
    This function crops an image to the center of the image. The resulting image will be at least as large as
    min_width x min_height, unless the image is smaller than that.

    :param img: The image to crop.
    :param min_width: The minimum width of the cropped image.
    :param min_height: The minimum height of the cropped image.
    """
    # Get image dimensions
    h, w = img.shape[:2]

    # Calculate crop coordinates
    start_y = max((h - min_height) // 2, 0)
    start_x = max((w - min_width) // 2, 0)
    end_y = start_y + min_height
    end_x = start_x + min_width

    # Crop, ensuring we stay within image bounds
    cropped_img = img[start_y:min(end_y, h), start_x:min(end_x, w)]
    return cropped_img