from typing import Iterable, Union

import numpy as np

from deepfake_detection.analysis.utils import centercrop
from deepfake_detection.data import ImageInstance, FileImageInstance


def prnu_fstv(instance: ImageInstance) -> np.ndarray:
    """
    This function extracts the PRNU pattern from a given image instance using the
    2nd order First Step Total Variation (FSTV) method (https://doi.org/10.1016/j.diin.2013.08.002).

    :param instance: Instance of FileImageInstance to extract PRNU from.
    :return: ImageInstance containing the extracted PRNU.
    """
    # Get image data (ensure float32 for numerical precision)
    u0 = np.array(instance.data).astype(np.float32)

    # Compute gradients
    grad_x = np.gradient(u0, axis=1)
    grad_y = np.gradient(u0, axis=0)

    # Calculate the strength of the gradients (add small epsilon to avoid taking the square root of 0)
    epsilon = 1e-8
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2 + epsilon ** 2)

    # Normalize gradients
    norm_grad_x = grad_x / grad_mag
    norm_grad_y = grad_y / grad_mag

    # Compute 2nd order gradients (gradient of normalized gradients)
    div_norm_x = np.gradient(norm_grad_x, axis=1)
    div_norm_y = np.gradient(norm_grad_y, axis=0)
    divergence = div_norm_x + div_norm_y

    # The noise is then defined as the negative divergence
    noise = -divergence

    return noise

def prnu_from_images(instances: Iterable[Union[ImageInstance, FileImageInstance]]) -> np.ndarray:
    """
    Calculates the mean PRNU pattern from a series of image instances.
    Currently, all individual PRNU images are centercropped to the width and height of the smallest image in the dataset.

    :param instances: Iterable of instances of ImageInstance or FileImageInstance to calculate the mean PRNU from.
    :return: numpy array containing the mean PRNU pattern.
    """
    # Get smallest width and height
    min_width = np.min([i.data.width for i in instances])
    min_height = np.min([i.data.height for i in instances])

    # Extract PRNU from all instances and average
    prnu_pattern = np.mean([centercrop(prnu_fstv(i), min_width=min_width, min_height=min_height) for i in instances],
                           axis=0)

    return prnu_pattern