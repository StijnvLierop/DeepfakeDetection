from typing import Union

import numpy as np
from PIL import Image

from deepfake_detection.data import ImageInstance, FileImageInstance
from deepfake_detection.analysis.utils import normalize_img


def fft(instance: Union[ImageInstance, FileImageInstance],
        channel: str = None,
        normalize: bool = False) -> ImageInstance:
    """
    Performs a Fast Fourier Transform (FFT) on an image and extracts the
    magnitude spectrum for visualization. The resulting spectrum is normalized
    to a pixel intensity range of 0-255.

    :param instance: An instance of ImageInstance containing the image data to be transformed.
    :param channel: A specific channel of the input image to be processed. Can be 'red', 'green', 'blue', or 'alpha'.
                    If not provided, the image is converted to grayscale before calculating the FFT.
    :param normalize: Whether to normalize the magnitude spectrum to the range 0-255 (mainly useful for visualization).
                      If False, the spectrum is returned in the range 0-255.
    :return: A new ImageInstance object containing the transformed image (magnitude spectrum).
    """
    # Extract image and transform from default PIL RGB color space to grayscale
    img = instance.data

    # If channel is selected, make sure image has this channel available
    if channel in ['red', 'green', 'blue']:
        assert img.mode == 'RGB'
    elif channel == 'alpha':
        assert img.mode == 'RGBA'

    # If channel is selected, get corresponding channel
    if channel == 'red':
        img = img.split()[0]
    elif channel == 'green':
        img = img.split()[1]
    elif channel == 'blue':
        img = img.split()[2]
    elif channel == 'alpha':
        img = img.split()[3]
    # Otherwise convert the image to grayscale
    else:
        img = img.convert('L')

    # Fourier transform image
    f_transform = np.fft.fft2(img)

    # Shift spectrum so low frequencies are in the center
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Extract magnitude spectrum only and apply some transformations for visualization
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

    # Normalize the data to the range 0-255 (if enabled)
    if normalize:
        magnitude_spectrum = normalize_img(magnitude_spectrum)

    return ImageInstance(data=Image.fromarray(magnitude_spectrum.astype(np.uint8)),
                         label=instance.label)