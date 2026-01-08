from typing import Union, List, Optional, Any

import torch
from torchvision.transforms import v2

from deepfake_detection.data.instance import FileImageInstance, ImageInstance
from deepfake_detection.data.dataset import Dataset
from deepfake_detection.models import Model
from deepfake_detection.models import Prediction
from deepfake_detection.models.custom_networks.fatformer import CLIPModel


class FatFormer(Model):
    """
    Implementation of the FatFormer model by Liu et al. (2024).

    More info about the model can be found here: https://github.com/Michel-liu/FatFormer/tree/main.
    """

    def __init__(
        self, ckpt: Optional[str] = None, device: str = "cuda", name: str = "FatFormer"
    ):
        """
        :param: ckpt: Path to the checkpoint file of the CNNDetect model.
        :param device: Device to use for inference.
        """
        super().__init__(name=name)
        self.model = None
        self.ckpt = ckpt
        self.device = device

    def load_model(self):
        self.model = CLIPModel(name="ViT-L/14").to(self.device)
        checkpoint = torch.load(self.ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"], strict=False)

    def forward(self, inputs: Any) -> Any:
        # Pass through the model
        logits = self.model(inputs)

        # Return logits and (optionally) loss
        return {"logits": logits}

    def predict_batch(
        self, instances: Union[List[Union[ImageInstance, FileImageInstance]], Dataset]
    ) -> List[Prediction]:
        # If model not yet loaded, load model
        if self.model is None:
            self.load_model()
        self.model.eval()

        # Get model inputs
        transform_func = self.get_input_transform_func()
        model_inputs = torch.stack(
            [transform_func(i.data) for i in instances], dim=0
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.forward(model_inputs)["logits"]
            out = logits.sigmoid().flatten().tolist()

        # Transform to Prediction
        return [Prediction(classification={"fake": o, "real": 1 - o}) for o in out]

    @staticmethod
    def get_input_transform_func() -> v2.Compose:
        transforms = [
            v2.Resize(
                (256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=True
            ),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transforms = v2.Compose(transforms)
        return transforms
