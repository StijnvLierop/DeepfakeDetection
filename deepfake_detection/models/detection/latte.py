from collections import OrderedDict
from typing import Optional, Union, List, Any

import torch
from torchvision.transforms import v2

from deepfake_detection.data.instance import FileImageInstance, ImageInstance
from deepfake_detection.models import Model, Prediction
from deepfake_detection.models.model import TrainableMixin
from deepfake_detection.models.custom_networks.latte import LatentTrajectoryClassifier

_SD_MODEL_ID = "Manojb/stable-diffusion-2-1-base"
_TRACKED_TIMESTEPS = [981, 741, 521, 261, 1]


class Latte(TrainableMixin, Model):
    """
    Implementation of the Latte model by Vasilcoiu et al. (2025).

    More info about the model can be found here: https://github.com/AnaMVasilcoiu/LATTE-Diffusion-Detector.
    """

    def __init__(self,
                 ckpt: Optional[str] = None,
                 name: str = "Latte",
                 load_model: bool = True,
                 *args,
                 **kwargs):
        self.classifier = None
        self.ckpt = ckpt
        super().__init__(*args, **kwargs)
        Model.__init__(self, name=name, load_model=load_model)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def load_model(self):
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        from transformers import CLIPTokenizer, CLIPTextModel

        # Classifier — registered as an nn.Module submodule, included in checkpoints.
        self.classifier = LatentTrajectoryClassifier(clip_type="convnext_base_in22k", process_latents_separately=True)
        if self.ckpt:
            state_dict = torch.load(self.ckpt, weights_only=False, map_location="cpu")
            state_dict = state_dict["model_state_dict"]
            new_state_dict = OrderedDict(
                (k[7:] if k.startswith("module.") else k, v)
                for k, v in state_dict.items()
            )
            self.classifier.load_state_dict(new_state_dict)
        else:
            print("No checkpoint provided, initializing Latte classifier with random weights.")

        # Frozen SD 2.1 pipeline — stored via object.__setattr__ to bypass nn.Module's
        # submodule registration, keeping them out of state_dict() and checkpoints.
        vae = AutoencoderKL.from_pretrained(_SD_MODEL_ID, subfolder="vae").eval()
        unet = UNet2DConditionModel.from_pretrained(_SD_MODEL_ID, subfolder="unet").eval()
        tokenizer = CLIPTokenizer.from_pretrained(_SD_MODEL_ID, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(_SD_MODEL_ID, subfolder="text_encoder").eval()
        noise_scheduler = DDPMScheduler.from_pretrained(_SD_MODEL_ID, subfolder="scheduler")

        for m in (vae, unet, text_encoder):
            m.requires_grad_(False)

        object.__setattr__(self, '_vae', vae)
        object.__setattr__(self, '_unet', unet)
        object.__setattr__(self, '_tokenizer', tokenizer)
        object.__setattr__(self, '_text_encoder', text_encoder)
        object.__setattr__(self, '_noise_scheduler', noise_scheduler)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # Move non-registered diffusion components alongside the classifier.
        if self.classifier is not None:
            device = next(self.classifier.parameters()).device
            for attr in ('_vae', '_unet', '_text_encoder'):
                comp = getattr(self, attr, None)
                if comp is not None:
                    object.__setattr__(self, attr, comp.to(device))
        return result

    def extract_latent_trajectories(self, inputs: torch.Tensor) -> torch.Tensor:
        device = inputs.device
        latents = (
            self._vae.encode(inputs).latent_dist.sample()
            * self._vae.config.scaling_factor
        )
        latent_sequences = []
        for t in _TRACKED_TIMESTEPS:
            noise = torch.randn_like(latents)
            timesteps = torch.full((latents.shape[0],), t, device=device, dtype=torch.long)
            t_scalar = torch.tensor(t, device=device, dtype=torch.long)
            noisy_latents = self._noise_scheduler.add_noise(latents, noise, timesteps)
            text_inputs = self._tokenizer(
                ["a photo"],
                max_length=self._tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            # Expand to batch size: the UNet cross-attention derives its internal batch
            # dimension from encoder_hidden_states, so a batch-1 prompt causes the
            # attention output to be [1, B*H*W, d] instead of [B, H*W, d], breaking
            # the residual add.
            encoder_hidden_states = self._text_encoder(text_inputs["input_ids"])[0]
            encoder_hidden_states = encoder_hidden_states.expand(latents.shape[0], -1, -1)
            model_pred = self._unet(noisy_latents, timesteps, encoder_hidden_states).sample
            latent_sequences.append(
                self._noise_scheduler.step(model_pred, t_scalar, noisy_latents).prev_sample
            )
        return torch.stack(latent_sequences, dim=1)  # [B, T, 4, 32, 32]

    def forward(self, inputs: Any, labels: Any = None, **kwargs) -> Any:
        with torch.no_grad():
            # Disable autocast: the SD 2.1 UNet attention produces shape mismatches
            # under fp16 autocast (residual add between tensors of size 3136 vs 784).
            with torch.amp.autocast(device_type='cuda', enabled=False):
                features = self.extract_latent_trajectories(inputs.float())
        # Cast back so the classifier runs normally under the Trainer's autocast context.
        features = features.to(inputs.dtype)
        logits, _ = self.classifier(inputs, features)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {
            "loss": loss,
            "logits": logits,
            "output": torch.softmax(logits, dim=1)[:, 1],
        }

    def predict_batch(self, instances: List[Union[ImageInstance, FileImageInstance]]) -> List[Prediction]:
        self.classifier.eval()
        device = next(self.classifier.parameters()).device
        model_inputs = torch.stack(
            [self.transform_input(i) for i in instances], dim=0
        ).to(device)
        with torch.no_grad():
            features = self.extract_latent_trajectories(model_inputs)
            logits, _ = self.classifier(model_inputs, features)
            probs = torch.softmax(logits, dim=1).cpu().numpy().tolist()
        return [
            Prediction(classification={"fake": float(prob[0]), "real": float(prob[1])})
            for prob in probs
        ]

    @staticmethod
    def transform_input(instance: ImageInstance) -> torch.Tensor:
        img = instance.data.convert("RGB") if hasattr(instance.data, 'convert') else instance.data
        return v2.Compose([
            v2.Resize((224, 224), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])(img)
