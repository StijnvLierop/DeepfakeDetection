from typing import Optional, Union, List

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import SegformerForSemanticSegmentation

from deepfake_detection.data.instance import ImageInstance, FileImageInstance
from deepfake_detection.models.model import TrainableMixin
from deepfake_detection.models import Model, Prediction


class SegFormerDetector(TrainableMixin, Model):
    """
    SegFormer-based binary segmentation model for predicting forged regions in images.

    Uses a HuggingFace SegformerForSemanticSegmentation backbone pretrained on ImageNet.

    During training the dataset must supply a ``"masks"`` tensor of shape ``(H, W)``
    with values 0 (authentic) or 1 (forged).
    """

    def __init__(
        self,
        pretrained_model_name: str = "nvidia/mit-b2",
        num_labels: int = 2,
        size: int = 512,
        ckpt: Optional[str] = None,
        name: str = "SegFormerForgeryDetector",
        load_model: bool = True,
        *args,
        **kwargs,
    ):
        self.pretrained_model_name = pretrained_model_name
        self.num_labels = num_labels
        self.size = size
        self.ckpt = ckpt
        self.model = None
        super().__init__(*args, **kwargs)
        Model.__init__(self, name=name, load_model=load_model)

    def load_model(self):
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.pretrained_model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
            id2label={0: "authentic", 1: "forged"},
            label2id={"authentic": 0, "forged": 1},
        )
        if self.ckpt is not None:
            state_dict = torch.load(self.ckpt, map_location="cpu", weights_only=True)
            if any(k.startswith("model.") for k in state_dict):
                state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)

    def forward(self, inputs, masks=None, **kwargs):
        outputs = self.model(pixel_values=inputs, labels=masks)
        upsampled = F.interpolate(
            outputs.logits,
            size=(self.size, self.size),
            mode="bilinear",
            align_corners=False,
        )
        return {"loss": outputs.loss, "logits": upsampled}

    def predict_batch(
        self, instances: List[Union[ImageInstance, FileImageInstance]]
    ) -> List[Prediction]:
        device = next(self.model.parameters()).device
        predictions = []
        for instance in instances:
            tensor = self.transform_input(instance).unsqueeze(0).to(device)
            with torch.no_grad():
                out = self.forward(tensor)
            prob_map = out["logits"].softmax(dim=1)[0, 1].cpu().numpy()
            predictions.append(Prediction(images={"forgery mask": prob_map}))
        return predictions

    def transform_input(self, instance, size: Optional[int] = None) -> torch.Tensor:
        size = size or self.size
        img = instance.data.convert("RGB") if hasattr(instance.data, "convert") else instance.data
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((size, size), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(img)
