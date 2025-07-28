import numpy as np
from PIL import Image, ImageFilter
import itertools

from deepfake_detection.data import ImageInstance


def noise_residual(instance: ImageInstance, mode: str = None) -> ImageInstance:
    """
    This function calculates the noise residual of a given image instance using a median filtering step.

    :param instance: An instance of ImageInstance containing the image data to be transformed.
    :param mode: A specific channel or mode of the input image to be processed.
                 Can be 'red', 'green', 'blue', 'alpha' or 'gray'.
                 If not provided, the noise residual of all channels is calculated and output in a single image.
    :return: A new ImageInstance object containing the noise residual.
    """
    # Get image
    img = instance.data

    # If channel is selected, make sure image has this channel available
    if mode in ['red', 'green', 'blue']:
        assert img.mode == 'RGB'
    elif mode == 'alpha':
        assert img.mode == 'RGBA'

    # If channel is selected, get corresponding channel
    if mode == 'red':
        img = img.split()[0]
    elif mode == 'green':
        img = img.split()[1]
    elif mode == 'blue':
        img = img.split()[2]
    elif mode == 'alpha':
        img = img.split()[3]
    elif mode == 'gray':
        img = img.convert('L')

    # Apply Median filter to smooth the image
    blurred = img.filter(ImageFilter.MedianFilter())

    # Convert images to NumPy arrays
    image_np = np.array(img, dtype=np.float32)
    blurred_np = np.array(blurred, dtype=np.float32)

    # Calculate noise residual
    noise_residual = blurred_np - image_np

    return ImageInstance(data=Image.fromarray(noise_residual.astype(np.uint8)),
                         label=instance.label)


def channel_noise_imbalance_ratio(instance: ImageInstance):
    """
    This function calculates the channel noise imbalance ratio (CNIR) for a given image instance.
    The CNIR quantifies the balance between the noise in different image channels. This balance might deviate
    from real images for certain generative models and therefore could be a useful feature.

    :param instance: An instance of ImageInstance.
    :return: The CNIR value for the given image instance.
    """

    # Calculate noise residual for each channel
    residuals = []
    for c in instance.data.split():
        residuals.append(np.array(noise_residual(ImageInstance(data=c)).data))

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

