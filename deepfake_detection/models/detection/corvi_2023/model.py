import os

import torch
from torch import load
from torchvision.transforms import transforms, Compose
from .resnet_mod import resnet50

from deepfake_detection.data.datasets.instance import ImageInstance
from deepfake_detection.models.model import Model
from ...prediction import Prediction


class Corvi2023Model(Model):
    """
    Implementation of Corvi 2023 (https://doi.org/10.48550/arXiv.2211.00680).
    Code heavily based on the implementation found here: https://grip-unina.github.io/ClipBased-SyntheticImageDetection/

    In the original paper an image is classified as synthetic when the score output by the model is bigger than 0.

    :param weights_dir: Path to the folder containing the weights of the model.
    :param device: Which device to use for computations.
    """

    def __init__(self, weights_dir: str, device: str='cuda:0'):
        super(Corvi2023Model, self).__init__(name='Corvi2023')
        self.weights_dir = weights_dir
        self.device = device
        self.model = None

    def load_model(self):
        model_path = os.path.join(self.weights_dir, 'weights.pth')
        model = resnet50(num_classes=1, stride0=1, dropout=0.5)
        dat = load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(dat['model'])
        self.model = model.to(self.device).eval()

    def predict(self, instance: ImageInstance) -> Prediction:
        # Load model if not yet loaded
        if self.model is None:
            self.load_model()

        # Define image transformation
        transform = Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                            )

        # Run inference
        out_tens = self.model(torch.stack([transform(instance.data)], 0)
                              .clone().to(self.device)).cpu().detach().numpy()

        # Transform output
        return Prediction(classification={'score':out_tens[0, 0]})