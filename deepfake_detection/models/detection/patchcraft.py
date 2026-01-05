from typing import Union, List, Optional, Any

import PIL
import torch
from torchvision.transforms import v2

from deepfake_detection.data import FileImageInstance, Dataset
from deepfake_detection.data import ImageInstance
from deepfake_detection.models import Model
from deepfake_detection.models import Prediction
from deepfake_detection.models.custom_networks.patchcraft import PatchCraftNetwork, ED


class PatchCraft(Model):
    """
    Implementation of the PatchCraft model by Zhong et al. (2023).

    More info about the model can be found here: https://github.com/Ekko-zn/AIGCDetectBenchmark.
    """

    def __init__(self, ckpt: Optional[str] = None, device: str = 'cuda', name: str = 'PatchCraft'):
        """
        :param: ckpt: Path to the checkpoint file of the CNNDetect model.
        :param device: Device to use for inference.
        """
        super().__init__(name=name)
        self.model = None
        self.ckpt = ckpt
        self.device = device

        # Define loss function for training
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def load_model(self):
        # Load architecture
        self.model = PatchCraftNetwork().to(self.device)

        # Get weights
        if self.ckpt:

            # Load state dict
            state_dict = torch.load(self.ckpt,
                                    weights_only=True,
                                    map_location='cpu')['netC']

            # Check if keys are prefixed with "model." and remove these prefixes if present
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            # Load weights in model
            self.model.load_state_dict(state_dict)

        else:
            print("No checkpoint provided, initializing model with random weights.")

    def predict_batch(self, instances: Union[List[Union[ImageInstance, FileImageInstance]], Dataset])\
            -> List[Prediction]:

        # If model not yet loaded, load model
        if self.model is None:
            self.load_model()

        # Set model to eval mode for inference
        self.model.eval()

        # Transform instances to tensor
        model_inputs = torch.stack([self.transform_inputs(i.data) for i in instances], dim=0).to(self.device)

        # Run inference
        with torch.no_grad():
            out = self.forward(model_inputs)['logits'].sigmoid().flatten().tolist()

        # Transform to Prediction
        return [Prediction(classification={'fake': o, 'real': 1 - o}) for o in out]

    def forward(self, inputs: Any, labels: Any = None, **kwargs) -> Any:
        # Run forward pass
        logits = self.model(inputs)

        # If labels given, calculate loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1), labels.float())

        # Return logits and (optionally) loss
        return {'loss': loss,
                'logits': logits}

    @staticmethod
    def transform_inputs(img: PIL.Image, num_patches: int = 3) -> torch.Tensor:
        """
        Preprocess image for PatchCraft model.

        It breaks down the image into random crops and then reorganizes them into two new composite images based on
        their texture complexity (spatial gradients).

        :param img: Image to preprocess.
        :param num_patches: Scaling factor for the number of patches in the image to look at.
        :return: Preprocessed image as a tensor to pass to the model.
        """

        num_block = int(pow(2, num_patches))
        patchsize = int(224 / num_block)
        randomcrop = v2.RandomCrop(patchsize)

        minsize = min(img.size)
        if minsize < patchsize:
            img = v2.Resize((patchsize, patchsize))(img)

        img = v2.ToTensor()(img)

        imgori = img.clone().unsqueeze(0)
        img_template = torch.zeros(3, 224, 224)
        img_crops = []
        for i in range(num_block * num_block * 3):
            cropped_img = randomcrop(img)
            texture_rich = ED(cropped_img)
            img_crops.append([cropped_img, texture_rich])

        img_crops = sorted(img_crops, key=lambda x: x[1])

        count = 0
        for ii in range(num_block):
            for jj in range(num_block):
                img_template[:, ii * patchsize:(ii + 1) * patchsize, jj * patchsize:(jj + 1) * patchsize] = \
                img_crops[count][0]
                count += 1
        img_poor = img_template.clone().unsqueeze(0)

        count = -1
        for ii in range(num_block):
            for jj in range(num_block):
                img_template[:, ii * patchsize:(ii + 1) * patchsize, jj * patchsize:(jj + 1) * patchsize] = \
                img_crops[count][0]
                count -= 1
        img_rich = img_template.clone().unsqueeze(0)
        img = torch.cat((img_poor, img_rich), 0)
        return img