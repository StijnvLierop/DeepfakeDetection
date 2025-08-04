import numpy as np


def fft(img: np.ndarray) -> np.ndarray:
    """
    Performs a Fast Fourier Transform (FFT) on an image and extracts the magnitude spectrum for visualization.

    :param img: A numpy array containing the image data to be transformed of the form
                (height, width) or (height, width, channels). Note that taking the fft of all color channels
                individually is different from taking the fft of a 3d image once.
    :return: A numpy array containing the transformed image (magnitude spectrum) of the form (height, width, channels).
    """
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