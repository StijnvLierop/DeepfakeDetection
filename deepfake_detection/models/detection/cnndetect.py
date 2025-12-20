from typing import Union, List, Optional, Any

import torch
from torchvision.transforms import v2

from deepfake_detection.data import FileImageInstance, Dataset
from deepfake_detection.data import ImageInstance
from deepfake_detection.models import Model
from deepfake_detection.models.model import TrainableMixin
from deepfake_detection.models.networks.resnet_cnndetect import resnet50
from deepfake_detection.models import Prediction


def process_input(instance: Union[ImageInstance, FileImageInstance]) -> torch.Tensor:
    cpu_transforms = v2.Compose([
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return cpu_transforms(instance.data)


class CNNDetect(Model, TrainableMixin):
    """
    Implementation of the CNNDetect model by Peter Wang et al. (2020).

    More info about the model can be found here: https://github.com/PeterWang512/CNNDetection/tree/master.
    """

    def __init__(self, ckpt: Optional[str] = None, device: str = 'cuda'):
        """
        :param: ckpt: Path to the checkpoint file of the CNNDetect model.
        :param device: Device to use for inference.
        """
        super(CNNDetect, self).__init__(name='CNNDetect')
        self.model = None
        self.ckpt = ckpt
        self.device = device

    def load_model(self):
        # Load architecture
        self.model = resnet50(num_classes=1).to(self.device)

        # Load weights
        if self.ckpt:
            state_dict = torch.load(self.ckpt,
                                    weights_only=True,
                                    map_location='cpu')
            self.model.load_state_dict(state_dict)
        else:
            print("No checkpoint provided, initializing model with random weights.")

    def predict_batch(self, instances: Union[List[Union[ImageInstance, FileImageInstance]], Dataset])\
            -> List[Prediction]:

        # If model not yet loaded, load model
        if self.model is None:
            self.load_model()

        # Set model to eval mode for inference
        self.model.eval()

        # Transform instances to tensor
        model_inputs = torch.stack([process_input(i) for i in instances], dim=0).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(model_inputs)["logits"]
            out = logits.sigmoid().flatten().tolist()

        # Transform to Prediction
        return [Prediction(classification={'fake': o, 'real': 1 - o}) for o in out]

    def get_model_parameters(self) -> Any:
        if self.model is None:
            self.load_model()
        return self.model.parameters()

    def forward_pass(self, inputs: Any) -> Any:
        return self.model(inputs)['logits']

    def save_weights(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def train(self) -> None:
        self.model.train()
