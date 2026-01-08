from typing import Union, List, Optional, Any

import torch
from torchvision.transforms import v2
from safetensors.torch import load_file

from deepfake_detection.data import FileImageInstance, Dataset
from deepfake_detection.data import ImageInstance
from deepfake_detection.models import Model
from deepfake_detection.models.model import TrainableMixin
from deepfake_detection.models import Prediction
from deepfake_detection.models.custom_networks.freqnet import FreqNetCNN


class FreqNet(TrainableMixin, Model):
    """
    Implementation of the FreqNet model by Tan et al. (2024).

    More info about the model can be found here: https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection.
    """

    def __init__(
        self, ckpt: Optional[str] = None, device: str = "cuda", *args, **kwargs
    ):
        """
        :param: ckpt: Path to the checkpoint file of the CNNDetect model.
        :param device: Device to use for inference.
        """
        super().__init__(*args, **kwargs)
        self.model = None
        self.ckpt = ckpt
        self.device = device

        # Define loss function for training
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def load_model(self):
        # Load architecture
        self.model = FreqNetCNN().to(self.device)

        # Get weights
        if self.ckpt:
            # For loading newly trained models
            if self.ckpt.endswith(".safetensors"):
                state_dict = load_file(self.ckpt)

                # Check if keys are prefixed with "model." and remove these prefixes if present
                if any(k.startswith("model.") for k in state_dict.keys()):
                    state_dict = {
                        k.replace("model.", ""): v for k, v in state_dict.items()
                    }

            # For loading old model files
            else:
                state_dict = torch.load(
                    self.ckpt, weights_only=True, map_location="cpu"
                )

            # Load weights in model
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError:
                self.model.load_state_dict(state_dict["model"])

        else:
            print("No checkpoint provided, initializing model with random weights.")

    def predict_batch(
        self, instances: Union[List[Union[ImageInstance, FileImageInstance]], Dataset]
    ) -> List[Prediction]:
        # If model not yet loaded, load model
        if self.model is None:
            self.load_model()

        # Set model to eval mode for inference
        self.model.eval()

        # Get transform func
        transform_func = self.get_input_transform_func(resize=True, crop=True)

        # Transform instances to tensor
        model_inputs = torch.stack(
            [transform_func(i.data) for i in instances], dim=0
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.forward(model_inputs)["logits"]
            out = logits.sigmoid().flatten().tolist()

        # Transform to Prediction
        return [Prediction(classification={"fake": o, "real": 1 - o}) for o in out]

    def forward(self, inputs: Any, labels: Any = None, **kwargs) -> Any:
        # Run forward pass
        logits = self.model(inputs)

        # If labels given, calculate loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1), labels.float())

        # Return logits and (optionally) loss
        return {"loss": loss, "logits": logits}

    @staticmethod
    def get_input_transform_func(resize: bool = False, crop: bool = True) -> v2.Compose:
        # Define base transforms
        transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        # Add optional transformations
        if crop:
            transforms.insert(0, v2.CenterCrop(224))
        if resize:
            transforms.insert(
                0,
                v2.Resize(
                    (256, 256),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                    max_size=None,
                ),
            )

        # Compose all transformations
        transforms = v2.Compose(transforms)
        return transforms
