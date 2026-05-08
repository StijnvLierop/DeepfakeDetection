import numpy as np


def prnu_fstv(img_array: np.ndarray) -> np.ndarray:
    """
    This function extracts the PRNU pattern from a given image instance using the
    2nd order First Step Total Variation (FSTV) method (https://doi.org/10.1016/j.diin.2013.08.002).

    :param img_array: Image array to extract PRNU from.
    :return: ImageInstance containing the extracted PRNU.
    """
    # Get image data (ensure float32 for numerical precision)
    u0 = img_array.astype(np.float32)

    # Compute gradients
    grad_x = np.gradient(u0, axis=1)
    grad_y = np.gradient(u0, axis=0)

    # Calculate the strength of the gradients (add small epsilon to avoid taking the square root of 0)
    epsilon = 1e-8
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + epsilon**2)

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
