import numpy as np
import scipy
from PIL import ImageFilter
import itertools

from PIL import Image


def noise_residual(img: np.array, image_filter: str='median') -> np.ndarray:
    """
    This function calculates the noise residual of a given image.

    :param img: A numpy array containing the image data to be transformed.
    :param image_filter: The filter to use for denoising the image. Must be one of:
                         - 'median': applies a Median filter.
                         - 'laplace': applies a Laplace filter.
    :return: A numpy array containing the noise residual.
    """
    # Apply filter to get denoised image
    if image_filter == 'median':
        denoised_img = np.array(Image.fromarray(img).filter(ImageFilter.MedianFilter()), dtype=np.float32)
    elif image_filter == 'laplace':
        denoised_img = scipy.ndimage.filters.laplace(np.array(img, dtype=np.float32))
    else:
        raise ValueError("Invalid filter. Must be one of: 'median' or 'laplace'.")

    # Calculate noise residual
    residual = denoised_img - img

    return residual


def channel_noise_imbalance_ratio(img: np.ndarray) -> float:
    """
    This function calculates the channel noise imbalance ratio (CNIR) for a given image.
    The CNIR quantifies the balance between the noise in different image channels. This balance might deviate
    from real images for certain generative models and therefore could be a useful feature.

    :param img: An numpy array containing the image data of shape (height, width, channels).
    :return: The CNIR value for the given image.
    """
    # Ensure the image has a channel dimension
    if img.ndim != 3:
        raise ValueError("Image must have a channel dimension. "
                         "Please ensure the provide image has shape (height, width, channels).")

    # Calculate noise residual for each channel
    residuals = []
    for c in range(img.shape[2]):
        residuals.append(noise_residual(img[:, :, c]))

    # Calculate difference of all channel combinations
    pairs = itertools.combinations(residuals, 2)
    diffs = []
    for p1, p2 in pairs:
        diffs.append(np.mean(np.abs(p1.astype(float) - p2.astype(float))))

    # Calculate mean and standard deviation of noise
    mu_noise = np.mean(np.array(diffs))
    imbalance = np.std(np.array(diffs))

    # Calculate CNIR by dividing the standard deviation of the noise between all channel combinations
    # by the mean noise of all channels
    cnir = imbalance / mu_noise

    return cnir

