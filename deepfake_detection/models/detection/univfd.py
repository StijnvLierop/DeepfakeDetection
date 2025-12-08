from typing import Union, List

import torch
from torchvision.transforms import v2

from deepfake_detection.data import ImageInstance, FileImageInstance
from deepfake_detection.data import Dataset
from deepfake_detection.models import Model
from deepfake_detection.models import Prediction
from deepfake_detection.models.networks.clip import clip


def process_input(instance: Union[ImageInstance, FileImageInstance]) -> torch.Tensor:
    transformations = [
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                      std=[0.26862954, 0.26130258, 0.27577711])
    ]
    transform = v2.Compose(transformations)
    return transform(instance.data)


class UnivFD(Model):
    """
    Implementation of the UniversalFakeDetect (UnivFD) model by Ojha et al. (2023).

    More info about the model can be found here: https://github.com/WisconsinAIVision/UniversalFakeDetect.
    """

    def __init__(self, ckpt: str, device: str = 'cuda'):
        super().__init__("UnivFD")
        self.model = None
        self.fc = None
        self.ckpt = ckpt
        self.device = device


    def load_model(self):
        self.model, _ = clip.load("ViT-L/14", device="cpu")
        self.model.to(self.device)
        self.fc = torch.nn.Linear(768, 1).to(self.device)
        state_dict = torch.load(self.ckpt, map_location='cpu', weights_only=True)
        self.fc.load_state_dict(state_dict)
        self.model.eval()


    def forward(self, x):
        features = self.model.encode_image(x)
        return self.fc(features)


    def predict(self, instance: Union[ImageInstance, FileImageInstance]) -> Prediction:

        # If model not yet loaded, load model
        if self.model is None:
            self.load_model()

        # Get model inputs
        model_inputs = process_input(instance).to(self.device).unsqueeze(0)

        # Make predictions
        with torch.no_grad():
            logits = self.forward(model_inputs)
            out = logits.sigmoid().flatten().tolist()
            return Prediction(classification={'fake': out[0], 'real': 1 - out[0]})


    def predict_batch(self, instances: Union[List[Union[ImageInstance, FileImageInstance]], Dataset])\
            -> List[Prediction]:

        # If model not yet loaded, load model
        if self.model is None:
            self.load_model()

        # Get model inputs
        model_inputs = torch.stack([process_input(i) for i in instances], dim=0).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.forward(model_inputs)
            out = logits.sigmoid().flatten().tolist()

        # Transform to Prediction
        return [Prediction(classification={'fake': o, 'real': 1 - o}) for o in out]