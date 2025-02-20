import os

import numpy as np
import torch
from PIL.Image import Image

from deepfake_detection.data.datasets.instance import ImageInstance
from deepfake_detection.models.detection.DIF_2020.DCNN import DnCNN
from deepfake_detection.models.model import Model
from deepfake_detection.models.prediction import Prediction


class DIFModel(Model):
    """
    Based on https://github.com/Sergo2020/DIF_pytorch_official.
    """

    def __init__(self, model_dir: str, device='cuda:0'):
        super().__init__(name='DIR')
        self.model_dir = model_dir
        self.fingerprints_dir = os.path.join(model_dir, 'fingerprints')
        self.device = device
        self.denoiser = None
        self.fingerprints = None

    def load_model(self):
        # Initialize denoiser
        denoiser_prnu_np = np.load(os.path.join(self.model_dir, r"clean_real.npy"), allow_pickle=True)
        denoiser_prnu = torch.tensor(denoiser_prnu_np.transpose((2, 0, 1))).to(self.device).unsqueeze(0)
        self.denoiser = DnCNN(3, 4).to(self.device)
        self.denoiser.prnu = denoiser_prnu

        # Load model fingerprints
        self.fingerprints = {}
        for generator in os.listdir(self.fingerprints_dir):
            # Get fingerprint path
            fingerprint_path = os.path.join(self.fingerprints_dir, generator, 'chk_0.pt')

            # Load fingerprint
            data_dict = torch.load(fingerprint_path)

            # Store camera1 and fake mu
            self.fingerprints[generator] = {'fingerprint': data_dict['Fingerprint'],
                                            'mu_real': np.mean(data_dict['Train Real'][-20:]),
                                            'mu_fake': np.mean(data_dict['Train Fake'][-20:])}

    def predict(self, instance: ImageInstance) -> Prediction:

        # If denoiser and fingerprints still none, load model
        if not self.denoiser or not self.fingerprints:
            self.load_model()

        # Crop image
        img = instance.data
        w, h = img.size
        left = (w - 256) / 2
        top = (h - 256) / 2
        right = (w + 256) / 2
        bottom = (h + 256) / 2
        img = np.array(img.crop((left, top, right, bottom)))
        img_tensor = torch.tensor(img.transpose((2, 0, 1))).type(torch.float32).div(255)
        img_tensor = img_tensor.unsqueeze(0)

        # Denoise image and
        residual = self.denoiser.denoise(img_tensor.to(self.device)).float()

        # Calculate corr for every generator
        real_distances = []
        distances = {}
        for generator in self.fingerprints.keys():
            # Get fingerprint
            generator_fingerprint = self.fingerprints[generator]['fingerprint']

            # Calculate correlation between fingerprint and noise residual
            corr = self.corr_fun(generator_fingerprint, residual)
            corr = corr.mean((1, 2, 3)).cpu().detach().numpy()

            # Calculate distance between class medians
            distance = self.distance(corr,
                                     self.fingerprints[generator]['mu_real'],
                                     self.fingerprints[generator]['mu_fake']
                                     )

            # Add to dict
            distances[generator] = 1 - float(distance[0, 1])
            real_distances.append(1 - float(distance[0, 0]))

        distances['real'] = min(real_distances)

        return Prediction(classification=distances)

    def corr_fun(self, out, target):
        # Pearson Correlation Coefficient (NNC(0,0))
        out = self.norm_val(out)
        target = self.norm_val(target)

        return out * target

    def norm_val(self, arr):
        return (arr - arr.mean((1, 2, 3)).view(-1, 1, 1, 1)) / (arr.std((1, 2, 3)).view(-1, 1, 1, 1) + 1e-8)

    def distance(self, arr, mu_a, mu_b):
        dist_arr2a = np.sqrt(((arr - mu_a) ** 2)).reshape((-1, 1))
        dist_arr2b = np.sqrt(((arr - mu_b) ** 2)).reshape((-1, 1))
        return np.concatenate((dist_arr2a, dist_arr2b), axis=1)