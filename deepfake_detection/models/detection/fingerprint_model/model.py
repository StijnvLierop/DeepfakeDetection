import os
from typing import Sequence, Union, Optional

import numpy as np

from analysis.prnu import prnu_fstv
from deepfake_detection.analysis.utils import average_over_images, centercrop
from deepfake_detection.data import FileImageInstance, ImageInstance
from deepfake_detection.models import Model, Prediction


class ImageFingerprintModel(Model):
    """
    This model uses the average frequency spectrum of the noise residual of a set of images for detection.
    The model compares the normalized cross-correlation between an image and a library of fingerprints of known models.

    :param fingerprint_dir: Directory to store fingerprints in / load fingerprints from.
    """

    def __init__(self, fingerprint_dir: str='fingerprints'):
        super().__init__(name='ImageFingerprintModel')
        self.fingerprint_dir = fingerprint_dir
        self.fingerprints = None
        self.fingerprint_size = (256, 256)


    def load_model(self):
        # Load fingerprints
        self.fingerprints = {}
        for f in os.listdir(self.fingerprint_dir):
            f_name = f.split('.')[0].split('/')[-1]
            model = f_name.split("_")[0]
            self.fingerprints[model] = np.load(os.path.join(self.fingerprint_dir, f))


    def extract_fingerprint(self,
                            instances: Sequence[Union[ImageInstance, FileImageInstance]],
                            dataset_name: Optional[str]=None,
                            save_fingerprint: bool=True) -> np.ndarray:
        """
        Extract a model fingerprint by averaging the frequency spectrum of the noise residual of a set of images.
        The fingerprint will be the size of the smallest image in the provided instance set.

        :param instances: The instances to compute the fingerprint of. These should all be from a single, known model.
        :param dataset_name: The name of the dataset the instances belong to.
                             If provided, this is used to name the saved fingerprint.
        :param save_fingerprint: Whether to save the fingerprint to the configured directory or not.
        """

        # Check if all instances have the same source
        sources = set([i.annotation.source_label for i in instances])
        if len(sources) > 1 or None in sources:
            raise ValueError("All instances must have the same known source which "
                             "cannot be None to estimate a meaningful fingerprint.")

        # Estimate model fingerprint
        fingerprint = average_over_images(instances,
                                          lambda x: prnu_fstv(centercrop(x,
                                                                          self.fingerprint_size[0],
                                                                          self.fingerprint_size[1]),
                                                             ),
                                          verbose=True)

        # Save fingerprint
        source = sources.pop()
        f_name = os.path.join(self.fingerprint_dir, source + ("_" + dataset_name if dataset_name else ""))
        if save_fingerprint:
            if not os.path.exists(self.fingerprint_dir):
                os.mkdir(self.fingerprint_dir)
            np.save(f"{f_name}.npy", fingerprint)

        # Store in dictionary
        if self.fingerprints is None:
            self.fingerprints = {}
        self.fingerprints[source] = np.array(fingerprint)

        return fingerprint


    def predict(self, instance: Union[ImageInstance, FileImageInstance]) -> Prediction:
        """
        Calculates the PCE of the instance with the known fingerprints of different models.
        """

        # Store results in dict
        results = {}

        # Load model (if not yet loaded)
        if self.fingerprints is None:
            self.load_model()

        # For each fingerprint
        for model, model_fingerprint in self.fingerprints.items():

            # Calculate PCE of fingerprint with instance (resize image to match fingerprint)
            img_array = prnu_fstv(np.array(instance.data))
            img_array = centercrop(img_array, model_fingerprint.shape[0], model_fingerprint.shape[1])
            results[model] = self.normalized_cross_correlation(model_fingerprint, np.array(img_array))

        # Return prediction
        return Prediction(classification=results)


    def pce(self, model_fingerprint: np.ndarray, img_fingerprint: np.ndarray) -> float:
        """
        Calculate peak to correlation energy (PCE) of two fingerprints of the same size.

        :param model_fingerprint: Fingerprint of a model of size (H, W, C) or (H, W).
        :param img_fingerprint: Fingerprint of an image of size (H, W, C) or (H, W).
        :return pce: PCE of the fingerprints.
        """

        # Compute cross-correlation using FFT
        cross_power = np.fft.fft2(model_fingerprint) * np.fft.fft2(img_fingerprint).conj()
        corr = np.fft.ifft2(cross_power)
        corr_abs = np.abs(corr)

        # Find peak value
        peak = np.max(corr_abs)
        # Compute total correlation energy
        energy = np.sum(corr_abs ** 2)
        # Compute PCE value
        pce = (peak ** 2) / (energy / corr.size)

        return float(pce)


    def normalized_cross_correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate normalized cross correlation between two arrays.

        :param a: First array.
        :param b: Second array.
        :return: Normalized cross correlation between a and b.
        """
        a = a - np.mean(a)
        b = b - np.mean(b)
        numerator = np.sum(a * b)
        denominator = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
        return float(numerator / denominator) if denominator != 0 else 0.0