import logging
from typing import Sequence, Union, Callable

import numpy as np
from tqdm import tqdm

from deepfake_detection.data import FileImageInstance, ImageInstance


def average_over_images(instances: Sequence[Union[ImageInstance, FileImageInstance]],
                        func: Callable[..., np.ndarray],
                        verbose: bool=False) -> np.ndarray:
    """
    This function processes a sequence of image instances, applies a specified function on the image data, and computes
    the average result. It ensures that the images are cropped to the smallest dimensions (width and height) present in
    the input sequence before averaging. The input sequence must contain at least one image instance to compute the mean.

    :param instances: A sequence of instances of ImageInstance or FileImageInstance to process.
    :param func: The function to apply to each image instance.
    :param verbose: Whether to show a progress bar while processing the images. Defaults to False.
    :return: The average result of applying the function to the image data of each instance in the input sequence.
    """
    # Make sure that length of sequence is at least one
    if len(instances) == 0:
        raise IndexError("Cannot calculate average from empty sequence.")

    # Get smallest width and height
    if verbose:
        logging.info("Calculating minimum width and height...")
    min_width = np.min([i.data.width for i in instances])
    min_height = np.min([i.data.height for i in instances])

    # Take the mean result of all processed instances
    result = np.mean(
        [centercrop(func(np.array(i.data)),
                    min_width=min_width,
                    min_height=min_height) for i in
         (tqdm(instances, desc="Processing images") if verbose else instances)],
        axis=0)

    return result


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