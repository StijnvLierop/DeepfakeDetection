from typing import Sequence, Union, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deepfake_detection.analysis.frequency import fft
from deepfake_detection.analysis.prnu import prnu_fstv
from deepfake_detection.analysis.utils import average_over_images, normalize_img
from deepfake_detection.data.dataset import Dataset
from deepfake_detection.data.instance import ImageInstance


def isolate_generative_traces(instances: Union[Sequence[ImageInstance], Dataset],
                              real_spectrum: Optional[str] = None,
                              cmap: str = 'plasma') -> ImageInstance:
    """
    Isolates the generative traces from a generator by taking the average frequency spectrum of the noise residuals
    of the given instances. It optionally subtracts the average spectrum from a real image dataset to remove
    low-frequency image information.

    :param instances: A sequence of instances from which to extract the common traces.
    :param real_spectrum: The path to a file containing the average frequency spectrum of the real images.
                          If provided, this will override any real_instances provided.
    :param cmap: The colormap to use for visualizing the traces. Defaults to 'plasma'.
    """

    # Apply relevant filters to isolate the generative traces
    traces = avg_fft_of_noise_residual(instances)

    # Optionally, substract the average spectrum from the real images to better visualize the traces
    if real_spectrum:
        real_spectrum = np.load(real_spectrum)
        traces -= real_spectrum

    # Convert to image and apply a color map
    cmap = plt.get_cmap(cmap).reversed()
    gray_traces = Image.fromarray((traces * 255).astype(np.uint8)).convert('L')
    colored_traces = ImageInstance(data=Image.fromarray((cmap(np.array(gray_traces))[:, :, :3] * 255).astype(np.uint8)))

    return colored_traces


def avg_fft_of_noise_residual(instances: Union[Sequence[ImageInstance], Dataset]) -> np.ndarray:
    """
    This function computes the average FFT of the PRNU noise residual of a sequence of image instances.

    :param instances: A sequence of instances.
    :return: The average FFT of the PRNU noise residual.
    """
    extracted = average_over_images(instances, lambda img: fft(prnu_fstv(img))).astype(np.uint8)
    return extracted