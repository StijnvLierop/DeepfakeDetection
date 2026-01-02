from typing import Union, List, Any

import torch
from torchvision.transforms import v2

from deepfake_detection.data import ImageInstance, FileImageInstance
from deepfake_detection.data import Dataset
from deepfake_detection.models import Model
from deepfake_detection.models import Prediction
from deepfake_detection.models.networks.clip import clip
from models.model import TrainableMixin


class UnivFD(TrainableMixin, Model):
    """
    Implementation of the UniversalFakeDetect (UnivFD) model by Ojha et al. (2023).

    More info about the model can be found here: https://github.com/WisconsinAIVision/UniversalFakeDetect.
    """

    def __init__(self, ckpt: str, device: str = 'cuda', name: str = 'UnivFD', *args, **kwargs):
        Model.__init__(self, name)
        super().__init__(*args, **kwargs)
        self.clip_encoder = None
        self.fc = None
        self.ckpt = ckpt
        self.device = device

        # Define loss function for training
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def load_model(self):
        # Load clip encoder
        self.clip_encoder, _ = clip.load("ViT-L/14", device="cpu")
        self.clip_encoder.to(self.device)
        self.clip_encoder.requires_grad_(False)

        # Load fully connected layer
        self.fc = torch.nn.Linear(768, 1).to(self.device)
        state_dict = torch.load(self.ckpt, map_location='cpu', weights_only=True)
        self.fc.load_state_dict(state_dict)

    def forward(self, inputs: Any, labels: Any = None, **kwargs) -> Any:
        # Get CLIP features
        features = self.clip_encoder.encode_image(inputs)

        # Pass through fully connected layer
        logits = self.fc(features)

        # If labels given, calculate loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1), labels.float())

        # Return logits and (optionally) loss
        return {'loss': loss,
                'logits': logits}

    def predict_batch(self, instances: Union[List[Union[ImageInstance, FileImageInstance]], Dataset])\
            -> List[Prediction]:

        # If model not yet loaded, load model
        if self.clip_encoder is None:
            self.load_model()
        self.clip_encoder.eval()

        # Get model inputs
        transform_func = self.get_input_transform_func(resize=False)
        model_inputs = torch.stack([transform_func(i.data) for i in instances], dim=0).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.forward(model_inputs)['logits']
            out = logits.sigmoid().flatten().tolist()

        # Transform to Prediction
        return [Prediction(classification={'fake': o, 'real': 1 - o}) for o in out]

    @staticmethod
    def get_input_transform_func(resize: bool = False) -> v2.Compose:
        transforms = [
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
        ]
        if resize:
            transforms.insert(0, v2.Resize(256,
                                           interpolation=v2.InterpolationMode.BILINEAR,
                                           antialias=True)
                              )
        transforms = v2.Compose(transforms)
        return transforms