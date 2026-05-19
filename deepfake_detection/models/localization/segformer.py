from typing import Optional, Union, List

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
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
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):
        self.pretrained_model_name = pretrained_model_name
        self.num_labels = num_labels
        self.size = size
        self.ckpt = ckpt
        self.gradient_checkpointing = gradient_checkpointing
        self.model = None
        self._input_transform = None
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
        if self.gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
            except ValueError:
                import warnings

                warnings.warn(
                    "SegFormer does not support gradient checkpointing; ignoring."
                )
                self.gradient_checkpointing = False
        if self.ckpt is not None:
            if self.ckpt.endswith(".safetensors"):
                state_dict = load_safetensors(self.ckpt)
            else:
                try:
                    state_dict = torch.load(
                        self.ckpt, map_location="cpu", weights_only=True
                    )
                except Exception:
                    # Checkpoint contains non-tensor objects (e.g. saved with optimizer state);
                    # weights_only=False is safe for locally-produced checkpoints.
                    state_dict = torch.load(
                        self.ckpt, map_location="cpu", weights_only=False
                    )
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            if any(k.startswith("model.") for k in state_dict):
                state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        self._resize_transform = self._build_resize_transform(self.size)
        self._normalize_transform = self._build_normalize_transform()
        self._input_transform = self._build_transform(self.size)

    def _build_resize_transform(self, size: int) -> v2.Compose:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(
                    (size, size),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
            ]
        )

    def _build_normalize_transform(self) -> v2.Compose:
        return v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _build_transform(self, size: int) -> v2.Compose:
        return v2.Compose(
            [
                *self._build_resize_transform(size).transforms,
                *self._build_normalize_transform().transforms,
            ]
        )

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
        # Get model device
        device = next(self.model.parameters()).device
        was_training = self.model.training

        # Set model to eval mode
        self.model.eval()
        try:
            # Transform inputs to tensor
            batch = torch.stack([self.transform_input(i) for i in instances]).to(device)

            # Run prediction
            with torch.no_grad():
                out = self.forward(batch)
            prob_maps = out["logits"].softmax(dim=1)[:, 1].cpu().numpy()
            return [Prediction(images={"forgery mask": p}) for p in prob_maps]
        finally:
            if was_training:
                self.model.train()

    def transform_input(self, instance, size: Optional[int] = None) -> torch.Tensor:
        size = size or self.size
        img = (
            instance.data.convert("RGB")
            if hasattr(instance.data, "convert")
            else instance.data
        )
        transform = (
            self._input_transform
            if (size == self.size and self._input_transform is not None)
            else self._build_transform(size)
        )
        return transform(img)
