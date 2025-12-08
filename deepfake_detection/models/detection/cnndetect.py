from typing import Union

import torch
from torchvision import transforms

from data import FileImageInstance
from deepfake_detection.data import ImageInstance
from deepfake_detection.models import Model
from deepfake_detection.models.networks.resnet import resnet50
from deepfake_detection.models import Prediction


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


class CNNDetect(Model):
    """
    Implementation of the CNNDetect model by Peter Wang et al. (2020).

    More info about the model can be found here: https://github.com/PeterWang512/CNNDetection/tree/master.
    """

    def __init__(self, ckpt: str):
        """
        param: ckpt: Path to the checkpoint file of the CNNDetect model.
        """
        super(CNNDetect, self).__init__(name='CNNDetect')
        self.model = None
        self.ckpt = ckpt


    def load_model(self):
        self.model = resnet50(num_classes=1)
        self.load_weights(self.ckpt)


    def load_weights(self, ckpt):
        state_dict = torch.load(ckpt, weights_only=True, map_location='cpu')
        try:
            self.model.load_state_dict(state_dict['model'])
        except:
            self.model.load_state_dict(state_dict)


    def process_input(self, instance: Union[ImageInstance, FileImageInstance]) -> torch.Tensor:
        # Define preprocessing transformations
        transformations = [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN['imagenet'], std=STD['imagenet'])
        ]
        transform = transforms.Compose(transformations)
        return transform(instance.data)


    def predict(self, instance: Union[ImageInstance, FileImageInstance]) -> Prediction:

        # If model not yet loaded, load model
        if self.model is None:
            self.load_model()

        # Transform instance to tensor
        model_input = self.process_input(instance).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            logits = self.model(model_input)["logits"]
            out = logits.sigmoid().flatten().tolist()

        # Transform to Prediction
        return Prediction(classification={'fake': out[0], 'real': 1-out[0]})
