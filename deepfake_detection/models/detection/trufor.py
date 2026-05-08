from typing import Optional, Union, List, Any

import torch
from torchvision.transforms import v2

from deepfake_detection.data.instance import FileImageInstance, ImageInstance
from deepfake_detection.models import Model, Prediction
from deepfake_detection.models.model import TrainableMixin
from deepfake_detection.models.custom_networks.trufor_models.cmx.builder_np_conf import (
    EncoderDecoder,
)
from deepfake_detection.models.custom_networks.trufor_models.config import (
    default_config,
)


class TruFor(TrainableMixin, Model):
    """
    Implementation of the TruFor model by Guillaro et al. (2023).

    More info about the model can be found here: https://github.com/grip-unina/TruFor/tree/main.
    """

    def __init__(
        self,
        ckpt: Optional[str] = None,
        name: str = "TruFor",
        load_model: bool = True,
        *args,
        **kwargs,
    ):
        self.model = None
        self.ckpt = ckpt
        super().__init__(*args, **kwargs)
        Model.__init__(self, name=name, load_model=load_model)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def load_model(self):
        self.model = EncoderDecoder(cfg=default_config())
        if self.ckpt is not None:
            state_dict = torch.load(self.ckpt, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state_dict["state_dict"])
        else:
            print("No checkpoint provided, initializing model with random weights.")

    def forward(self, inputs: Any, labels: Any = None, **kwargs) -> Any:
        _, conf, det, _ = self.model(inputs)
        loss = None
        if labels is not None:
            loss = self.loss_fn(det.view(-1), labels.float())
        return {
            "loss": loss,
            "logits": det,
            "output": det.sigmoid().flatten(),
        }

    def predict_batch(
        self, instances: List[Union[ImageInstance, FileImageInstance]]
    ) -> List[Prediction]:
        device = next(self.model.parameters()).device
        predictions = []
        for instance in instances:
            item = self.transform_input(instance).unsqueeze(0).to(device)
            with torch.no_grad():
                pred, conf, det, _ = self.model(item)
                tamper_map = torch.softmax(pred[0], dim=0)[1].cpu().numpy()
                confidence_map = (
                    torch.sigmoid(conf[0][0]).cpu().numpy()
                    if conf is not None
                    else None
                )
                detection_score = torch.sigmoid(det).item() if det is not None else None
            predictions.append(
                Prediction(
                    classification={"manipulated": detection_score},
                    images={"tamper map": tamper_map, "confidence map": confidence_map},
                )
            )
        return predictions

    @staticmethod
    def transform_input(instance: ImageInstance, size: int = 256) -> torch.Tensor:
        img = (
            instance.data.convert("RGB")
            if hasattr(instance.data, "convert")
            else instance.data
        )
        transforms = v2.Compose(
            [
                v2.Resize(
                    (size, size),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )
        return transforms(img) / 256.0
