from typing import Union, List

import torch
from torchvision.transforms import v2

from deepfake_detection.data import ImageInstance, FileImageInstance, Dataset
from deepfake_detection.models import Prediction
from deepfake_detection.models.model import Model
from deepfake_detection.models.networks.resnet_npr import resnet50


def process_input(instance: Union[ImageInstance, FileImageInstance]) -> torch.Tensor:
    cpu_transforms = v2.Compose([
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return cpu_transforms(instance.data)


class NPR(Model):

    def __init__(self, ckpt: str, device: str = 'cuda'):
        super(NPR, self).__init__("NPR")
        self.model = None
        self.ckpt = ckpt
        self.device = device


    def load_model(self):
        self.model = resnet50(num_classes=1).to(self.device)
        self.load_weights(self.ckpt)
        self.model.eval()


    def load_weights(self, ckpt):
        state_dict = torch.load(ckpt, map_location='cpu')
        try:
            self.model.load_state_dict(state_dict['model'], strict=False)
        except:
            print('Loading failed, trying to load model without module')
            self.model.load_state_dict(state_dict)


    def predict(self, instance: Union[ImageInstance, FileImageInstance]) -> Prediction:

        # If model not yet loaded, load model
        if self.model is None:
            self.load_model()

        # Transform instance to tensor
        model_inputs = process_input(instance).to(self.device).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            logits = self.model(model_inputs)["logits"]
            out = logits.sigmoid().flatten().tolist()

        # Transform to Prediction
        return Prediction(classification={'fake': out[0], 'real': 1 - out[0]})


    def predict_batch(self,
                      instances: Union[List[Union[ImageInstance, FileImageInstance]], Dataset]) \
            -> List[Prediction]:

        # If model not yet loaded, load model
        if self.model is None:
            self.load_model()

        # Transform instance to tensor
        model_inputs = torch.stack([process_input(i) for i in instances], dim=0).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(model_inputs)["logits"]
            out = logits.sigmoid().flatten().tolist()

        # Transform to Prediction
        return [Prediction(classification={'fake': o, 'real': 1 - o}) for o in out]