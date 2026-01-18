from typing import Union, List, Optional, Any

import torch
from torchvision.transforms import v2
from safetensors.torch import load_file

from deepfake_detection.data.instance import FileImageInstance, ImageInstance
from deepfake_detection.data.dataset import Dataset
from deepfake_detection.models import Model
from deepfake_detection.models.model import TrainableMixin
from deepfake_detection.models.custom_networks.resnet_cnndetect import resnet50
from deepfake_detection.models import Prediction


class CNNDetect(TrainableMixin, Model):
    """
    Implementation of the CNNDetect model by Peter Wang et al. (2020).

    More info about the model can be found here: https://github.com/PeterWang512/CNNDetection/tree/master.
    """

    def __init__(
        self,
        ckpt: Optional[str] = None,
        device: str = "cuda",
        name: str = "CNNDetect",
        *args,
        **kwargs,
    ):
        """
        :param: ckpt: Path to the checkpoint file of the CNNDetect model.
        :param device: Device to use for inference.
        """
        Model.__init__(self, name)
        super().__init__(*args, **kwargs)
        self.model = None
        self.ckpt = ckpt
        self.device = device

        # Define loss function for training
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def load_model(self):
        # Load architecture
        self.model = resnet50(num_classes=1).to(self.device)

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
        transform_func = self.get_input_transform_func(resize=False)

        # Transform instances to tensor
        model_inputs = torch.stack(
            [transform_func(i.data) for i in instances], dim=0
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            out = self.forward(model_inputs)["logits"].sigmoid().flatten().tolist()

        # Transform to Prediction
        return [Prediction(classification={"fake": o, "real": 1 - o}) for o in out]

    def forward(self, inputs: Any, labels: Any = None, **kwargs) -> Any:
        # Run forward pass
        logits = self.model(inputs)["logits"]

        # If labels given, calculate loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1), labels.float())

        # Return logits and (optionally) loss
        return {"loss": loss, "logits": logits}

    @staticmethod
    def get_input_transform_func(resize: bool = False) -> v2.Compose:
        transforms = [
            v2.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            v2.ToImage(),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if resize:
            transforms.insert(
                1,
                v2.Resize(
                    256, interpolation=v2.InterpolationMode.BILINEAR, antialias=True
                ),
            )
        transforms = v2.Compose(transforms)
        return transforms
