from typing import Union, List, Sequence

import torch
from PIL.Image import Image
from torchvision import transforms, models
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torch.nn import functional as F
import numpy as np

from deepfake_detection.data import Dataset, FileImageInstance, ImageInstance
from deepfake_detection.models.model import Model
from deepfake_detection.models.prediction import Prediction


def process_images(images: Sequence[Image]) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    return torch.stack([preprocess(i) for i in images])


class FakeInversion(Model):
    """
    Implementation of the FakeInversion model by Cazenavette et al. (2024).

    More info about the model can be found here: https://fake-inversion.github.io.
    """

    def __init__(self, ckpt: str, device: str = 'cuda'):
        super().__init__("FakeInversion")
        self.classifier = None
        self.captioning = None
        self.embedding = None
        self.feature_extractor = None
        self.device = device
        self.ckpt = ckpt


    def load_model(self):
        # Define captioning model
        self.captioning = ImageCaptioning()

        # Define feature extractor model
        self.feature_extractor = FeatureExtractor()

        # Define classifier
        self.classifier = models.resnet50(pretrained=True)
        self.classifier.fc = torch.nn.Linear(self.classifier.fc.in_features, 2)
        state_dict = torch.load(self.ckpt, weights_only=True, map_location='cpu')
        self.classifier.load_state_dict(state_dict['model'])
        self.classifier.to(self.device).eval()


    def predict_batch(self, instances: Union[List[Union[ImageInstance, FileImageInstance]], Dataset]) -> List[Prediction]:
        if not self.classifier:
            self.load_model()

        # Preprocess images and convert to a single 4D Tensor [B, C, H, W]
        imgs = [i.data for i in instances]
        img_tensor = process_images(imgs).to(self.device)

        # Generate captions
        captions = self.captioning.get_captions(imgs)

        # Extract features
        latents, noises, reconstructed_images = self.feature_extractor.extract_features(img_tensor, captions)

        # Pass reconstructed images directly to the classifier
        outputs = self.classifier(reconstructed_images.float())

        # Apply softmax function
        class_predictions = torch.argmax(F.softmax(outputs, dim=1), dim=1)

        # Transfer to cpu
        class_predictions = class_predictions.cpu().detach().numpy()
        latents = latents.cpu().detach().numpy()
        reconstructed_images = (reconstructed_images.permute(0, 2, 3, 1).cpu().detach().numpy() * 255).astype(np.uint8)

        # Transform to prediction objects
        predictions = []
        for i in range(len(instances)):
            pred = Prediction(classification={'fake': float(class_predictions[i]),
                                              'real': 1 - float(class_predictions[i])
                                              },
                              embedding=latents[i],
                              text=captions[i],
                              image=reconstructed_images[i]
                              )
            predictions.append(pred)

        return predictions


# BLIP: Image Captioning
class ImageCaptioning:

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",
                                                                  use_safetensors=True).to(self.device)

    def get_captions(self, images: List[Image]):
        inputs = self.processor(images, return_tensors="pt").to(self.device)
        caption_ids = self.model.generate(**inputs, do_sample=False)
        captions = self.processor.batch_decode(caption_ids, skip_special_tokens=True)
        return captions


# Stable Diffusion Feature Extraction
class FeatureExtractor:

    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", device: str = 'cuda'):
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name,
                                                            use_safetensors=True,
                                                            torch_dtype=torch.float16).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.scheduler.set_timesteps(50)


    def extract_features(self, image_tensor: torch.Tensor, captions: List[str], seed: int = 42):
        """
        Performs the 'FakeInversion' logic:
        1. Encode image to latent.
        2. Noise latent to t=49.
        3. Predict noise using U-Net conditioned on text embedding.
        4. Reconstruct images from predicted noise.

        :param image_tensor: Tensor of shape [B, C, H, W] representing the input image.
        :param captions: List of length B containing the text prompts.
        :param seed: Seed to use for sampling.
        """

        # This object manages the random state locally without affecting global PyTorch state
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Ensure input is float16 for the pipeline
        image_tensor = image_tensor.to(dtype=torch.float16)

        with torch.no_grad():
            # Encode Images (VAE)
            latents = self.pipe.vae.encode(image_tensor).latent_dist.sample(generator=generator)
            latents = latents * self.pipe.vae.config.scaling_factor

            # Add Noise (Diffusion Process)
            # Simulate timestep 49 (near the end of the diffusion process)
            t_idx = 49
            # Create a tensor of shape [Batch] filled with 49
            timesteps = torch.full((latents.shape[0],), t_idx, device=self.device, dtype=torch.long)
            noise = torch.randn(latents.shape, generator=generator, device=self.device, dtype=latents.dtype)
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings of captions
            text_embeddings, _ = self.pipe.encode_prompt(
                prompt=captions,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )

            # Predict Noise (U-Net)
            # This detects the artifacts. Real images + Caption != Predicted Noise.
            # Fake images + Caption == Predicted Noise (roughly).
            noise_pred = self.pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample

            # Denoise / Reconstruct
            # Step back from t=49 to previous step using the model's prediction
            reconstructed_latents = self.scheduler.step(noise_pred, t_idx, noisy_latents).prev_sample

            # Decode (VAE)
            reconstructed_images = self.pipe.vae.decode(
                reconstructed_latents / self.pipe.vae.config.scaling_factor
            ).sample

        # Return outputs (keep on GPU for now, cast if needed later)
        return latents, noise, reconstructed_images
