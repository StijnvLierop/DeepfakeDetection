import numpy as np
import scipy
from matplotlib import pyplot as plt

from deepfake_detection.data.datasets.fileimagedataset import FileImageDataset
from skimage.restoration import denoise_wavelet


def extract_model_fingerprint(dataset: FileImageDataset) -> np.ndarray:
    """

    """
    # Transform images to arrays
    img_arrays = [np.array(i.data.crop((250, 0, 250, 0))) for i in dataset]

    # Pass all images through a denoiser and calculate residuals
    residuals = [denoise_wavelet(img) - img for img in img_arrays]

    # Take average of residuals
    avg_residual = np.mean(residuals, axis=0)

    # Transform to DCT space
    fingerprint = scipy.fftpack.rfft(avg_residual)

    plt.imshow(fingerprint)
    plt.show()

    return fingerprint