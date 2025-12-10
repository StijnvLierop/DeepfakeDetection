from typing import Union, List

import torch
from torchvision.transforms import v2
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler

from deepfake_detection.data import Instance, Dataset, ImageInstance, FileImageInstance
from deepfake_detection.models import Model, Prediction
from deepfake_detection.models.networks.latte import LatentTrajectoryClassifier


def instance_to_tensor(instance: Union[ImageInstance, FileImageInstance]) -> torch.Tensor:
    # Define transform
    default_transform = v2.Compose([
        v2.Resize(224, 224),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        v2.ToTensor(),
    ])
    return default_transform(instance.data)


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
        self.tracked_timesteps = "[981, 741, 521, 261, 1]"

        # Set models
        self.classifier = None
        self.vae = None
        self.unet = None
        self.tokenizer = None
        self.text_encoder = None
        self.noise_scheduler = None


    def load_model(self):
        # Load classifier
        self.classifier = LatentTrajectoryClassifier()
        state_dict = torch.load(self.ckpt, weights_only=True, map_location='cpu')
        self.classifier.load_state_dict(state_dict['model'])

        # Load diffusion model components to generate latent trajectories
        self.vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder="vae").to(self.device).eval()
        self.unet = UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder="unet").to(self.device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder="text_encoder").to(self.device).eval()
        self.noise_scheduler = DDPMScheduler.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder="scheduler")

        # Disable gradient computation for inference
        for model in (self.vae, self.unet, self.text_encoder):
            model.requires_grad_(False)


    def extract_latent_trajectory(self,
                       image: torch.Tensor)\
            -> torch.Tensor:

        # Transform instances to tensor
        # img_tensor = torch.stack([default_transform(i.data) for i in instances],
        #                          dim=0).to(self.device)

        # Extract latents
        latents = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor

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


    def predict(self, instance: Union[ImageInstance, FileImageInstance]) -> Prediction:

        if self.classifier is None:
            self.load_model()

        with torch.no_grad():

            # Convert instance to tensor
            img_tensor = instance_to_tensor(instance).to(self.device)

            # Extract features from instance
            features = self.extract_latent_trajectory(img_tensor).to(self.device)

            # Pass image and features to model
            logits, embeddings = self.classifier(img_tensor, features)

            # Convert logits to probabilities
            prob = torch.softmax(logits, dim=1)

            # Transform to prediction
            return Prediction(classification={"fake": float(prob[0]), "real": float(prob[1])})



    def predict_batch(self, instances: Union[List[Instance], Dataset]) -> List[Prediction]:
        pass