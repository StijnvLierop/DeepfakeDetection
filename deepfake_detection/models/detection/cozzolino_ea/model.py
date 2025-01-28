import numpy as np
from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode
import torchvision.transforms as transforms
import torch

from deepfake_detection.data.datasets.instance import ImageSequenceInstance, ImageInstance
from deepfake_detection.models.model import Model
from deepfake_detection.models.detection.cozzolino_ea.utils import create_architecture, load_weights, get_config

class Cozzolino2023Model(Model):

    # synthetic when bigger than 0

    def __init__(self, device='cuda:0'):
        super(Cozzolino2023Model, self).__init__(name='Cozzolino2023')
        _, model_path, arch, norm_type, patch_size = get_config('clipdet_latent10k_plus',
                                                                weights_dir='deepfake_detection/models/detection/'
                                                                            'cozzolino_ea/weights')
        model = load_weights(create_architecture(arch), model_path)
        self.model = model.to(device).eval()
        self.device = device

    def predict(self, instance: ImageInstance):

        # Define image transformation
        transform = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC),
                             CenterCrop((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                  std=(0.26862954, 0.26130258, 0.27577711),)
                             ])

        # Run inference
        out_tens = self.model(torch.stack([transform(instance.data)], 0)
                              .clone().to(self.device)).cpu().detach().numpy()

        # Transform output
        if out_tens.shape[1] == 1:
            out_tens = out_tens[:, 0]
        elif out_tens.shape[1] == 2:
            out_tens = out_tens[:, 1] - out_tens[:, 0]
        else:
            assert False

        if len(out_tens.shape) > 1:
            logit1 = np.mean(out_tens, (1, 2))
        else:
            logit1 = out_tens

        return logit1


class Cozzolino2023ModelImageSequence(Model):

    # synthetic when bigger than 0

    def __init__(self, device='cuda:0'):
        _, model_path, arch, norm_type, patch_size = get_config('clipdet_latent10k_plus',
                                                                weights_dir='deepfake_detection/models/detection/'
                                                                            'cozzolino_ea/weights')
        model = load_weights(create_architecture(arch), model_path)
        self.model = model.to(device).eval()
        self.device = device

    def predict(self, instance: ImageSequenceInstance):

        scores = []

        for img in instance.data:

            # Define image transformation
            transform = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC),
                                 CenterCrop((224, 224)), transforms.ToTensor(),
                                 transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                      std=(0.26862954, 0.26130258, 0.27577711),)
                                 ])

            # Run inference
            out_tens = self.model(torch.stack([transform(img.data)], 0)
                                  .clone().to(self.device)).cpu().detach().numpy()

            # Transform output
            if out_tens.shape[1] == 1:
                out_tens = out_tens[:, 0]
            elif out_tens.shape[1] == 2:
                out_tens = out_tens[:, 1] - out_tens[:, 0]
            else:
                assert False

            if len(out_tens.shape) > 1:
                logit1 = np.mean(out_tens, (1, 2))
            else:
                logit1 = out_tens

            scores.append(logit1)

        return np.min(scores)