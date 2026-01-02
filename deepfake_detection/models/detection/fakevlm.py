from typing import Union, List, Optional

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

from deepfake_detection.data import ImageInstance, FileImageInstance, Dataset
from deepfake_detection.models import Model
from deepfake_detection.models import Prediction


class FakeVLM(Model):
    """
    Implementation of the FakeVLM model by Wen et al. (2025).

    More info about the model can be found here: https://github.com/opendatalab/FakeVLM.
    """

    def __init__(self, device: str = 'cuda', model_path: Optional[str] = None):
        super().__init__("FakeVLM")
        self.model = None
        self.device = device
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = 'lingcco/fakeVLM'


    def load_model(self):
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf",
                                                       use_fast=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval().to(torch.device(self.device))


    def predict_batch(self, instances: Union[List[Union[ImageInstance, FileImageInstance]], Dataset])\
            -> List[Prediction]:
        # Load model when not yet loaded
        if self.model is None:
            self.load_model()

        # Define conversation template
        conversations = [[
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
        ] for _ in instances]

        # Process model input
        with torch.no_grad():
            prompts = self.processor.apply_chat_template(conversations, add_generation_prompt=True)
            model_input = self.processor(
                text=prompts,
                images=[instance.data for instance in instances],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            # Run inference
            out = self.model.generate(**model_input, max_new_tokens=200, do_sample=False)

            # Iterate through the batch to decode predictions
            predictions = []
            input_len = model_input.input_ids.shape[1]
            for i in range(len(instances)):
                # Extract the generated tokens for this specific instance in the batch
                generated_ids = out[i][input_len:]
                response = self.processor.decode(generated_ids, skip_special_tokens=True).split('?')[-1]

                # Logic to parse classification
                first_sentence = response.split('.')[0]
                if 'real' in first_sentence:
                    scores = {'fake': 0, 'real': 1}
                elif 'fake' in first_sentence:
                    scores = {'fake': 1, 'real': 0}
                else:
                    scores = {'fake': 0.5, 'real': 0.5}
                predictions.append(Prediction(classification=scores))

        return predictions