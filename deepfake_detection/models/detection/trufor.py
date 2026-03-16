from typing import Optional, Union, List

import numpy as np
import torch

from deepfake_detection.data.instance import FileImageInstance, ImageInstance
from deepfake_detection.models import Model, Prediction
from deepfake_detection.models.custom_networks.trufor_models.cmx.builder_np_conf import EncoderDecoder
from deepfake_detection.models.custom_networks.trufor_models.config import default_config


class TruFor(Model):
    """
    Implementation of the TruFor model by Guillaro et al. (2023).

    More info about the model can be found here: https://github.com/grip-unina/TruFor/tree/main.
    """
    def __init__(self,
                 ckpt: Optional[str] = None,
                 name: str = "TruFor",
                 device: str = "cuda"):
        self.device = device
        self.ckpt = ckpt

        # Define model
        self.model = EncoderDecoder(cfg=default_config()).to(self.device)

        super().__init__(name=name, load_model=True)


    def load_model(self):
        # If weights are provided, load them
        if self.ckpt is not None:
            state_dict = torch.load(self.ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict['state_dict'])
        else:
            print("No checkpoint provided, initializing model with random weights.")

    def predict_batch(self, instances: List[Union[ImageInstance, FileImageInstance]]) -> List[Prediction]:

        # If not yet loaded, load model
        if self.model is None:
            self.load_model()

        # Transform instances to tensors
        inputs = [self.transform_input(i) for i in instances]

        # Loop over inputs
        predictions = []
        for item in inputs:

            # Run inference
            with torch.no_grad():
                # Predicts
                pred, conf, det, _ = self.model(item.to(self.device))

                # Create tamper map
                tamper_map = torch.softmax(pred[0], dim=0)[1].cpu().numpy()

                # If returned, calculate a confidence map
                confidence_map: Optional[np.ndarray] = None
                if conf is not None:
                    confidence_map = torch.sigmoid(conf[0][0]).cpu().numpy()

                # If returned, calculate detection score
                detection_score: Optional[float] = None
                if det is not None:
                    detection_score = torch.sigmoid(det).item()

                # Add prediction to the list
                predictions.append(Prediction(classification={'manipulated': detection_score},
                                              images={'tamper map': tamper_map,
                                                      'confidence map': confidence_map})
                                   )
        return predictions

    @staticmethod
    def transform_input(instance: ImageInstance) -> torch.Tensor:
        rgb = np.asarray(instance.data.convert("RGB"), dtype=np.float32)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor / 256.0
        return tensor
