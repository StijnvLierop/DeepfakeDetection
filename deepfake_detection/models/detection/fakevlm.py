from typing import Union, List

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

from deepfake_detection.data import Instance, Dataset
from deepfake_detection.models import Model
from deepfake_detection.models import Prediction
from deepfake_detection.models.model import lazy_loader


class FakeVLM(Model):
    """
    Implementation of the FakeVLM model by Wen et al. (2025).

    More info about the model can be found here: https://github.com/opendatalab/FakeVLM.
    """

    def __init__(self, device: str = 'cuda'):
        super().__init__("FakeVLM")
        self.model = None
        self.device = device


    def load_model(self):
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf",
                                                       use_fast=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            'lingcco/fakeVLM',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval().to(torch.device(self.device))


    @lazy_loader
    def predict(self, instance: Instance) -> Prediction:

        # Define conversation template
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text",
                     "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Does the image looks real/fake?"},
                    {"type": "image"},
                ],
            },
        ]

        # Process model input
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        model_input = self.processor(
            text=prompt,
            images=instance.data,
            return_tensors="pt",
        ).to(self.device)

        # Run inference
        out = self.model.generate(**model_input, max_new_tokens=200, do_sample=False)

        # Decode response
        generated_ids = out[0][model_input.input_ids.shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Parse response to classification
        if 'real' in response.split('.')[0].lower():
            classification = {'fake': 0, 'real': 1}
        elif 'fake' in response.split('.')[0].lower():
            classification = {'fake': 1, 'real': 0}
        else:
            classification = {'fake': 0.5, 'real': 0.5}

        # Transform to Prediction
        return Prediction(classification=classification)


    def predict_batch(self, instances: Union[List[Instance], Dataset]) -> List[Prediction]:
        pass