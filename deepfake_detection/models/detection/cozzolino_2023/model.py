import os

from torch import load
from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode
import torchvision.transforms as transforms
import torch

from .openclipnet import OpenClipLinear
from deepfake_detection.data.datasets.instance import ImageInstance
from deepfake_detection.models.model import Model
from ...prediction import Prediction


class Cozzolino2023Model(Model):
    """
    Implementation of Cozzolino 2023 (https://doi.org/10.48550/arXiv.2312.00195).
    Code heavily based on the implementation found here: https://grip-unina.github.io/ClipBased-SyntheticImageDetection/

    In the original paper an image is classified as synthetic when the score output by the model is bigger than 0.

    Predictions are returned as classifications and the feature representations used as input for the classifier is
    returned as embedding.

    :param weights_dir: Path to the folder containing the weights of the model.
    :param device: Which device to use for computations.
    """

    def __init__(self, weights_dir: str, device='cuda:0'):
        super(Cozzolino2023Model, self).__init__(name='Cozzolino2023')
        self.weights_dir = weights_dir
        self.device = device
        self.model = None

    def load_model(self):
        model_path = os.path.join(self.weights_dir, 'weights.pth')
        model = OpenClipLinear(num_classes=1, pretrain='clipL14commonpool', normalize=True, next_to_last=True)
        dat = load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(dat['model'])
        self.model = model.to(self.device).eval()

    def preprocess(self, instance: ImageInstance):
        # Define image transformation
        transform = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC),
                             CenterCrop((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                  std=(0.26862954, 0.26130258, 0.27577711), )
                             ])

        return torch.stack([transform(instance.data)], 0)

    def predict(self, instance: ImageInstance) -> Prediction:
        # Load model if not yet loaded
        if self.model is None:
            self.load_model()

        # Run inference
        features = self.model.forward_features(self.preprocess(instance).clone().to(self.device))
        out = self.model.forward_head(features).cpu().detach().numpy()
        features = features.cpu().detach().numpy()

        # Transform output
        return Prediction(classification={'camera1': float(out[0, 0])<=0,
                                          'fake': float(out[0, 0])>0},
                          embedding=list(features[0].astype(float)))