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
        self,
        ckpt: Optional[str] = None,
        device: str = "cuda",
        name: str = "FatFormer",
        load_model: bool = True,
    ):
        """
        :param: ckpt: Path to the checkpoint file of the CNNDetect model.
        """
        self.model = None
        self.ckpt = ckpt
        self.device = device
        super().__init__(name=name, load_model=load_model)

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

        # Get model inputs
        model_inputs = torch.stack(
            [self.transform_input(i) for i in instances], dim=0
        ).to(next(self.model.parameters()).device)

        # Run inference
        with torch.no_grad():
            logits = self.forward(model_inputs)["logits"]
            out = logits.sigmoid().flatten().tolist()

        # Transform to Prediction
        return [Prediction(classification={"fake": o, "real": 1 - o}) for o in out]

    @staticmethod
    def transform_input(instance: ImageInstance) -> torch.Tensor:
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
        return transforms(instance.data)
