import torch
import torchvision

from deepfake_detection.models import Model, Prediction
from deepfake_detection.models.detection.chu2024fire.third_party.network_utils import FIRE_model

class FireModel(Model):
    """
    This class implements the FIRE model from Chu et al. (2024).

    @article{chu2024fire,
      title={FIRE: Robust Detection of Diffusion-Generated Images via Frequency-Guided Reconstruction Error},
      author={Chu, Beilin and Xu, Xuan and Wang, Xin and Zhang, Yufei and You, Weike and Zhou, Linna},
      journal={arXiv preprint arXiv:2412.07140},
      year={2024}
    }
    """

    def __init__(self, model_path: str, name):
        super().__init__(name)
        self.model_path = model_path
        self.model = None
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256, antialias=True),
                                                        torchvision.transforms.ToTensor()])

    def load_model(self):
        model = FIRE_model()
        model.load_state_dict(torch.load(self.model_path))
        model.cuda()
        self.model = model

    def predict(self, instance) -> Prediction:
        # Load model (if not yet loaded)
        if self.model is None:
            self.load_model()

        # Preprocess instance
        img = self.transform(instance.data).to(device='cuda').unsqueeze(0)

        # Make prediction
        out = self.model(img)[0]
        out = float(out.sigmoid().detach().cpu().numpy()[0][0])

        return Prediction(classification={'real': 1-out, 'fake': out})