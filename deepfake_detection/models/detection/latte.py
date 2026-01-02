from collections import OrderedDict
from typing import Union, List

import torch
from torchvision.transforms import v2
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

from deepfake_detection.data import Dataset, ImageInstance, FileImageInstance
from deepfake_detection.models import Model, Prediction
from deepfake_detection.models.custom_networks.latte import LatentTrajectoryClassifier


class Latte(Model):
    """
    Implementation of the Latte model by Vasilcoiu et al. (2025).

    More info about the model can be found here: https://github.com/AnaMVasilcoiu/LATTE-Diffusion-Detector.
    """

    def __init__(self, ckpt: str, device: str = 'cuda'):
        super().__init__("Latte")
        # Set params
        self.ckpt = ckpt
        self.device = device
        self.tracked_timesteps = [981, 741, 521, 261, 1]

        # Set models
        self.classifier = None
        self.vae = None
        self.unet = None
        self.tokenizer = None
        self.text_encoder = None
        self.noise_scheduler = None

    def load_model(self):
        # Load classifier
        self.load_classifier_model()

        # Load diffusion model components to generate latent trajectories
        self.vae = AutoencoderKL.from_pretrained('Manojb/stable-diffusion-2-1-base', subfolder="vae").to(self.device).eval()
        self.unet = UNet2DConditionModel.from_pretrained('Manojb/stable-diffusion-2-1-base', subfolder="unet").to(self.device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained('Manojb/stable-diffusion-2-1-base', subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained('Manojb/stable-diffusion-2-1-base', subfolder="text_encoder").to(self.device).eval()
        self.noise_scheduler = DDPMScheduler.from_pretrained('Manojb/stable-diffusion-2-1-base', subfolder="scheduler")

        # Disable gradient computation for inference
        for model in (self.vae, self.unet, self.text_encoder):
            model.requires_grad_(False)

    def load_classifier_model(self):
        # Load model
        self.classifier = LatentTrajectoryClassifier(clip_type="convnext_base_in22k")

        # Load weights
        state_dict = torch.load(self.ckpt, weights_only=False, map_location='cpu')

        # Load the original state dict
        state_dict = state_dict['model_state_dict']

        # Create a new state dict without the 'module.' prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove `module.`
            new_state_dict[name] = v

        # Now load the cleaned version
        self.classifier.load_state_dict(new_state_dict)
        self.classifier.to(self.device)

    def extract_latent_trajectories(self, inputs: torch.Tensor) -> torch.Tensor:

        # Extract latents
        latents = self.vae.encode(inputs).latent_dist.sample() * self.vae.config.scaling_factor

        # Create latent trajectories
        latent_sequences = []
        for t in self.tracked_timesteps:
            noise = torch.randn_like(latents)
            timesteps = torch.full((latents.shape[0],), t, device=latents.device, dtype=torch.long)

            # Apply noise and UNet prediction
            t_tensor_scheduler = torch.tensor(t, device=self.device, dtype=torch.long)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            prompts = ["a photo"]

            text_inputs = self.tokenizer(prompts,
                                    max_length=self.tokenizer.model_max_length,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt").to(self.device)
            encoder_hidden_states = self.text_encoder(text_inputs["input_ids"])[0]
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
            step_output = self.noise_scheduler.step(model_pred, t_tensor_scheduler, noisy_latents)
            current_latents = step_output.prev_sample
            latent_sequences.append(current_latents)

        # [B, T, 4, 32, 32]
        latent_sequences = torch.stack(latent_sequences, dim=1)

        return latent_sequences


    def predict_batch(self,  instances: Union[List[Union[ImageInstance, FileImageInstance]], Dataset]) -> List[Prediction]:
        # Load model when not yet loaded
        if self.classifier is None:
            self.load_model()

        # Set classifier to eval mode for inference
        self.classifier.eval()

        # Get transform func
        transform_func = self.get_input_transform_func()

        # Transform instances to tensor
        model_inputs = torch.stack([transform_func(i.data) for i in instances], dim=0).to(self.device)

        # Run inference
        with torch.no_grad():

            # Extract features from instance
            features = self.extract_latent_trajectories(model_inputs).to(self.device)

            # Pass image and features to model
            logits, embeddings = self.classifier(model_inputs, features)

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy().tolist()

        # Transform to predictions
        return [Prediction(classification={'fake': float(prob[0]), 'real': float(prob[1])}) for prob in probs]


    @staticmethod
    def get_input_transform_func() -> v2.Compose:
        transforms = [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
        transforms = v2.Compose(transforms)
        return transforms