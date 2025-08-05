import numpy as np
from scipy.signal import fftconvolve


def autocorrelation(img: np.ndarray) -> np.ndarray:
    """
    Computes the autocorrelation of an image.

    :param img: A numpy array containing the image data to be transformed of the form (height, width).
    :return: A numpy array containing the transformed image (autocorrelation) of the form (height, width).
    """
    # Check that image array is 2D
    if img.ndim != 2:
        raise ValueError("Image must be 2D.")

    # Subtract the mean to disregard differences between absolute pixel values
    image_zero_mean = img - np.mean(img)

    # Compute 2D autocorrelation using FFT-based convolution (correlate image with itself)
    autocorr = fftconvolve(image_zero_mean, image_zero_mean[::-1, ::-1], mode='full')

    return autocorr