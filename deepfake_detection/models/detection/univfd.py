from typing import Union, List, Any, Optional

import torch
from torchvision.transforms import v2
from clip import clip

from deepfake_detection.data.instance import FileImageInstance, ImageInstance
from deepfake_detection.data.dataset import Dataset
from deepfake_detection.models import Model
from deepfake_detection.models import Prediction
from deepfake_detection.models.model import TrainableMixin


class UnivFD(TrainableMixin, Model):
    """
    Implementation of the UniversalFakeDetect (UnivFD) model by Ojha et al. (2023).

    More info about the model can be found here: https://github.com/WisconsinAIVision/UniversalFakeDetect.
    """

    def __init__(
        self,
        ckpt: Optional[str] = None,
        name: str = "UnivFD",
        load_model: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.clip_encoder = None
        self.fc = torch.nn.Linear(768, 1)
        self.ckpt = ckpt

        Model.__init__(self, name=name, load_model=load_model)

        # Define loss function for training
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def load_model(self):
        # Load clip encoder
        self.clip_encoder, _ = clip.load("ViT-L/14", device="cpu")
        self.clip_encoder.requires_grad_(False)

        # If checkpoint is provided
        if self.ckpt:
            # Load state dict
            state_dict = torch.load(self.ckpt, map_location="cpu", weights_only=True)

            # Extract only the keys starting with "fc." and strip the prefix
            fc_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("fc."):
                    clean_key = k.replace("fc.", "")  # "fc.weight" -> "weight"
                    fc_state_dict[clean_key] = v

            # Load fully connected layer
            self.fc.load_state_dict(fc_state_dict)
        else:
            print("No checkpoint provided, initializing model with random weights.")
            torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)

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
        return {
            "loss": loss,
            "logits": logits,
            "embeddings": features,
            "out": logits.sigmoid().flatten(),
        }

    def predict_batch(
        self, instances: Union[List[Union[ImageInstance, FileImageInstance]], Dataset]
    ) -> List[Prediction]:

        # Get model inputs
        model_inputs = torch.stack(
            [self.transform_input(i, resize=False) for i in instances], dim=0
        ).to(next(self.fc.parameters()).device)

        # Run inference
        with torch.no_grad():
            out = self.forward(model_inputs)

        # Transform to Prediction
        return [
            Prediction(classification={"fake": out, "real": 1 - out}, embedding=embed)
            for out, embed in zip(
                out["out"].cpu().tolist(), out["embeddings"].cpu().tolist()
            )
        ]

    @staticmethod
    def transform_input(instance: ImageInstance, resize: bool = True) -> torch.Tensor:
        transforms = [
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
        if resize:
            transforms.insert(
                0,
                v2.Resize(
                    256, interpolation=v2.InterpolationMode.BILINEAR, antialias=True
                ),
            )
        transforms = v2.Compose(transforms)
        return transforms(instance.data)
