from typing import Union, List

import torch
from torchvision import transforms, models
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torch.nn import functional as F
import numpy as np

from deepfake_detection.data import Instance, Dataset, FileImageInstance, ImageInstance
from deepfake_detection.models.model import Model
from deepfake_detection.models.prediction import Prediction


def process_instance(instance: Union[FileImageInstance, ImageInstance]) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    return preprocess(instance.data)


class FakeInversion(Model):
    """
    Implementation of the FakeInversion model by Cazenavette et al. (2024).

    More info about the model can be found here: https://fake-inversion.github.io.
    """

    def __init__(self, device: str = 'cuda'):
        super().__init__("FakeInversion")
        self.classifier = None
        self.captioning = None
        self.embedding = None
        self.feature_extractor = None
        self.device = device


    def load_model(self):
        # Define captioning model
        self.captioning = ImageCaptioning()

        # Define embedding model
        self.embedding = TextEmbedding()

        # Define feature extractor model
        self.feature_extractor = FeatureExtractor()

        # Define classifier
        self.classifier = models.resnet50(pretrained=True)
        self.classifier.fc = torch.nn.Linear(self.classifier.fc.in_features, 2)  # Binary classification
        self.classifier.to(self.device).eval()


    def predict(self, instance: Union[ImageInstance, FileImageInstance]) -> Prediction:

        if not self.classifier:
            self.load_model()

        # Create img tensor
        img_tensor = process_instance(instance).to(self.device)

        # Generate caption
        caption = self.captioning.get_caption(instance.data)

        # Get text embedding
        text_embedding = self.embedding.get_embedding(caption)

        # Extract features
        latent, noise, reconstructed_image = self.feature_extractor.extract_features(img_tensor, text_embedding)
        reconstructed_image = reconstructed_image.squeeze(0)  # Remove batch dimension if present

        # Pass reconstructed_image directly to the classifier
        output = self.classifier(reconstructed_image.unsqueeze(0))

        # Transform to prediction object
        prediction = torch.argmax(F.softmax(output, dim=1), dim=1)

        return Prediction(classification={'fake': float(prediction[0]), 'real': 1 - float(prediction[0])},
                          embedding=latent.cpu().detach().numpy(),
                          text=caption,
                          image=(reconstructed_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                          )


    def predict_batch(self, instances: Union[List[Instance], Dataset]) -> List[Prediction]:
        pass



# BLIP: Image Captioning
class ImageCaptioning:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",
                                                                  use_safetensors=True).to(self.device)

    def get_caption(self, image):
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        caption_ids = self.model.generate(**inputs)
        caption = self.processor.decode(caption_ids[0], skip_special_tokens=True)
        return caption


# CLIP: Text Embedding
class TextEmbedding:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                                       use_safetensors=True)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                               use_safetensors=True).to(self.device)

    def get_embedding(self, caption):
        inputs = self.processor(text=[caption],
                                return_tensors="pt",
                                padding=True).to(self.device)
        text_embedding = self.model.get_text_features(**inputs)
        return text_embedding


# Stable Diffusion Feature Extraction
class FeatureExtractor:

    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", device: str = 'cuda'):
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name, use_safetensors=True).to(self.device)
        # The scheduler configuration is likely located under the model's directory.
        # Specify the 'scheduler' subfolder:
        self.scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        # Call set_timesteps to initialize num_inference_steps
        self.scheduler.set_timesteps(50) # You can adjust the number of steps here


    def extract_features(self, image, text_embedding):
        # Encode the image to latent space
        latents = self.pipe.vae.encode(image.unsqueeze(0).to(self.device)).latent_dist.sample()
        latents = latents * self.pipe.vae.config.scaling_factor

        # Invert using DDIM
        noise = torch.randn_like(latents).to(self.device)
        inverted_latents = self.scheduler.add_noise(latents,
                                                    noise,
                                                    torch.tensor([49], device=self.device, dtype=torch.long)
                                                    )

        # Reconstruct image from inverted latent
        with torch.no_grad():
            # Passing noise and inverted_latents as arguments and removing text_embedding as it is not the timestep
            reconstructed_latents = self.scheduler.step(noise, 49, inverted_latents).prev_sample
            reconstructed_image = self.pipe.vae.decode(reconstructed_latents / self.pipe.vae.config.scaling_factor).sample

        return latents, noise, reconstructed_image