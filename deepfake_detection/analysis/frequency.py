import numpy as np
from skimage.filters import window


def fft(img: np.ndarray, hamming_window: bool = False) -> np.ndarray:
    """
    Performs a Fast Fourier Transform (FFT) on an image and extracts the magnitude spectrum for visualization.

    :param img: A numpy array containing the image data to be transformed of the form
                (height, width) or (height, width, channels). Note that taking the fft of all color channels
                individually is different from taking the fft of a 3d image once.
    :param hamming_window: Whether to apply a Hamming window to the image before performing the FFT. This removes
                           the prominent horizontal and vertical frequency bands caused by the image edges due to
                           violating the periodicity assumption of the FFT.
    :return: A numpy array containing the transformed image (magnitude spectrum) of the form (height, width, channels).
    """
    # Apply Hamming window if set
    if hamming_window:
        # If color image
        if img.ndim == 3:
            # Apply window for every channel
            for channel in range(img.shape[2]):
                img[:, :, channel] = img[:, :, channel] * window(
                    "hamming", img.shape[:2]
                )
        # If grayscale image, apply once
        else:
            img = img * window("hamming", img.shape)

    # If array is 3D, transform so channel dimension comes first
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))

    # Fourier transform image
    f_transform = np.fft.fft2(img)

    # Shift spectrum so low frequencies are in the center
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Extract magnitude spectrum only and apply some transformations for visualization
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

    # If array is 3D, transpose back to normal image format (height, width, channels)
    if img.ndim == 3:
        magnitude_spectrum = np.transpose(magnitude_spectrum, (1, 2, 0))

    return magnitude_spectrum
