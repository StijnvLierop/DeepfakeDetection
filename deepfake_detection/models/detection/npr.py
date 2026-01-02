from typing import Union, List, Any
from collections import OrderedDict

import torch
from torchvision.transforms import v2

from deepfake_detection.data import ImageInstance, FileImageInstance, Dataset
from deepfake_detection.models import Prediction
from deepfake_detection.models.model import Model
from deepfake_detection.models.networks.resnet_npr import resnet50
from deepfake_detection.models.model import TrainableMixin
from models.training.augmentations.translate_duplicate import TranslateDuplicate


class NPR(TrainableMixin, Model):
    """
    Implementation of the Neighboring Pixel Relationships (NPR) model by Tan et al. (2023).

    More info about the model can be found here: https://github.com/chuangchuangtan/NPR-DeepfakeDetection.
    """

    def __init__(self, ckpt: str, device: str = 'cuda', name: str = 'NPR', *args, **kwargs):
        Model.__init__(self, name)
        super().__init__(*args, **kwargs)
        self.model = None
        self.ckpt = ckpt
        self.device = device
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def load_model(self):
        self.model = resnet50(num_classes=1).to(self.device)
        self.load_weights(self.ckpt)
        self.model.eval()

    def load_weights(self, ckpt):
        # Load state dict
        state_dict = torch.load(ckpt, map_location='cpu', weights_only=True)

        # Remove 'module.' prefix in state dict keys (if present)
        if 'model' in state_dict:
            new_state_dict = OrderedDict()
            for k, v in state_dict['model'].items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict

        # Load weights
        self.model.load_state_dict(new_state_dict, strict=True)

    def predict_batch(self,
                      instances: Union[List[Union[ImageInstance, FileImageInstance]], Dataset]) \
            -> List[Prediction]:

        # If model not yet loaded, load model
        if self.model is None:
            self.load_model()

        # Transform instance to tensor
        transform_func = self.get_input_transform_func(resize=False, crop=True, translate_and_duplicate=True)
        model_inputs = torch.stack([transform_func(i.data) for i in instances], dim=0).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.forward(model_inputs)['logits']
            out = logits.sigmoid().flatten().tolist()

        # Transform to Prediction
        return [Prediction(classification={'fake': o, 'real': 1 - o}) for o in out]

    def forward(self, inputs: Any, labels: Any = None, **kwargs) -> Any:
        # Run forward pass
        logits = self.model(inputs)

        # If labels given, calculate loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1), labels.float())

        # Return logits and (optionally) loss
        return {'loss': loss,
                'logits': logits}

    @staticmethod
    def get_input_transform_func(resize: bool = False,
                                 crop: bool = True,
                                 translate_and_duplicate: bool = False) -> v2.Compose:
        # Define base transforms
        transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        # Check if not both reisze and translate_and_duplicate are set to True
        if resize and translate_and_duplicate:
            raise ValueError("Cannot set both resize and translate_and_duplicate to True.")

        # Add optional transformations
        if crop:
            transforms.insert(0, v2.CenterCrop(224))
        if resize:
            transforms.insert(0, v2.Resize((256, 256),
                                           interpolation=v2.InterpolationMode.BILINEAR,
                                           antialias=True,
                                           max_size=None)
                              )
        elif translate_and_duplicate:
            transforms.insert(0, TranslateDuplicate(cropSize=224))

        # Compose all transformations
        transforms = v2.Compose(transforms)
        return transforms