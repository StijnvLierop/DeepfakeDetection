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


class CNNSpot(TrainableMixin, Model):
    """
    Implementation of the CNNSpot model by Peter Wang et al. (2020).

    More info about the model can be found here: https://github.com/PeterWang512/CNNDetection/tree/master.
    """

    def __init__(
        self,
        ckpt: Optional[str] = None,
        name: str = "CNNSpot",
        load_model: bool = True,
        *args,
        **kwargs,
    ):
        """
        :param: ckpt: Path to the checkpoint file of the CNNDetect model.
        """
        self.model = None
        self.ckpt = ckpt
        super().__init__(*args, **kwargs)
        Model.__init__(self, name=name, load_model=load_model)

        # Define loss function for training
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def load_model(self):
        # Load architecture
        self.model = resnet50(num_classes=1)

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

        # Transform instances to tensor
        model_inputs = torch.stack(
            [self.transform_input(i, resize=False) for i in instances], dim=0
        )

        # Move tensor to model device
        model_inputs = model_inputs.to(next(self.model.parameters()).device)

        # Run inference
        with torch.no_grad():
            out = self.forward(model_inputs)

        # Transform to Prediction
        return [Prediction(classification={"fake": float(output), "real": 1 - float(output)},
                           embedding=embedding.detach().cpu().numpy().flatten().tolist())
                for output, embedding in zip(out['output'], out['penultimate_layer'])]

    def forward(self, inputs: Any, labels: Any = None, **kwargs) -> Any:

        # Attach a hook to the penultimate layer of the model
        features = {}
        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook
        self.model.avgpool.register_forward_hook(get_features('penultimate'))

        # Run forward pass
        logits = self.model(inputs)["logits"]

        # If labels given, calculate loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1), labels.float())

        # Return logits and (optionally) loss
        return {"loss": loss,
                "logits": logits,
                "penultimate_layer": features['penultimate'],
                'output': logits.sigmoid().flatten()}

    @staticmethod
    def transform_input(instance: ImageInstance, resize: bool = False) -> torch.Tensor:
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
        return transforms(instance.data)
