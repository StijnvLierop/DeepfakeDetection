from typing import Union, List, Optional

import open_clip
import torch
from torch import nn
from torchvision import transforms

from deepfake_detection.data.instance import FileImageInstance, ImageInstance
from deepfake_detection.models import Model, TrainableMixin
from deepfake_detection.models import Prediction
from deepfake_detection.models.custom_networks.aide import ResNet, HPF, Bottleneck, Mlp, DCT_base_Rec_Module


class AIDE(TrainableMixin, Model):
    """
    Implementation of the AIDE model by Yan et al. (2025).

    More info about the model can be found here: https://github.com/shilinyan99/AIDE/tree/main.
    """

    def __init__(self,
                 ckpt: Optional[str] = None,
                 resnet_ckpt: Optional[str] = None,
                 convnext_ckpt: Optional[str] = None,
                 name: str = "AIDE",
                 device: str = "cuda",
                 *args,
                 **kwargs):
        Model.__init__(self, name=name)
        super().__init__(*args, **kwargs)
        self.device = device

        # Load checkpoints (ckpt is main checkpoint that should be loaded during inference)
        self.ckpt = ckpt
        self.resnet_ckpt = resnet_ckpt
        self.convnext_ckpt = convnext_ckpt

        # Initialize DCT module
        self.dct_module = DCT_base_Rec_Module().requires_grad_(False).to(self.device)

        # Define model layers
        self.hpf = HPF().to(self.device)
        self.model_min = ResNet(Bottleneck, [3, 4, 6, 3]).to(self.device)
        self.model_max = ResNet(Bottleneck, [3, 4, 6, 3]).to(self.device)
        self.fc = Mlp(2048 + 256, 1024, 2).to(self.device)
        self.openclip_convnext_xxl = None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        self.convnext_proj = nn.Sequential(
            nn.Linear(3072, 256),
        ).to(self.device)

    def forward(self, x):
        """
        Forward func for https://github.com/shilinyan99/AIDE/blob/main/models/AIDE.py.
        """
        x_minmin = x[:, 0]
        x_maxmax = x[:, 1]
        x_minmin1 = x[:, 2]
        x_maxmax1 = x[:, 3]
        tokens = x[:, 4]

        x_minmin = self.hpf(x_minmin)
        x_maxmax = self.hpf(x_maxmax)
        x_minmin1 = self.hpf(x_minmin1)
        x_maxmax1 = self.hpf(x_maxmax1)

        with torch.no_grad():
            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(tokens, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens, non_blocking=True).view(3, 1, 1)

            local_convnext_image_feats = self.openclip_convnext_xxl(
                tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            )  # [b, 3072, 8, 8]
            assert local_convnext_image_feats.size()[1:] == (3072, 8, 8)
            local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
            x_0 = self.convnext_proj(local_convnext_image_feats)

        x_min = self.model_min(x_minmin)
        x_max = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)

        x_1 = (x_min + x_max + x_min1 + x_max1) / 4
        x = torch.cat([x_0, x_1], dim=1)
        x = self.fc(x)

        return {'logits': x, 'output': torch.softmax(x, dim=1)}


    def load_model(self, ckpt: Optional[str] = None, resnet_ckpt: Optional[str] = None, convnext_ckpt: Optional[str] = None) -> None:

        # If resnet weights are provided, load them
        if resnet_ckpt is not None:
            pretrained_dict = torch.load(resnet_ckpt, map_location='cpu')

            model_min_dict = self.model_min.state_dict()
            model_max_dict = self.model_max.state_dict()

            for k in pretrained_dict.keys():
                if k in model_min_dict and pretrained_dict[k].size() == model_min_dict[k].size():
                    model_min_dict[k] = pretrained_dict[k]
                    model_max_dict[k] = pretrained_dict[k]

            self.model_min.load_state_dict(model_min_dict)
            self.model_max.load_state_dict(model_max_dict)

        # Load openclip model
        self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms("convnext_xxlarge",
                                                                                 pretrained=convnext_ckpt)
        self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
        self.openclip_convnext_xxl.head.global_pool = nn.Identity()
        self.openclip_convnext_xxl.head.flatten = nn.Identity()
        self.openclip_convnext_xxl.eval()
        self.openclip_convnext_xxl.to(self.device)

        # Turn off grads for convnext
        for param in self.openclip_convnext_xxl.parameters():
            param.requires_grad = False

        # Load weights
        if ckpt is not None:
            self.load_state_dict(torch.load(ckpt, map_location='cpu')['model'], strict=False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:

        transform_func = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct_module(x.to(self.device))

        x_0 = transform_func(x.to(self.device))
        x_minmin = transform_func(x_minmin)
        x_maxmax = transform_func(x_maxmax)
        x_minmin1 = transform_func(x_minmin1)
        x_maxmax1 = transform_func(x_maxmax1)

        input_tensor = torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0)

        return input_tensor


    def predict_batch(self, instances: List[Union[ImageInstance, FileImageInstance]]) -> List[Prediction]:

        # If model not yet loaded, load model
        if self.openclip_convnext_xxl is None:
            self.load_model(ckpt=self.ckpt)

        # Transform inputs
        to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        inputs = torch.stack([self.preprocess(to_tensor_transform(instance.data)) for instance in instances],
                             dim=0).to(self.device)

        # Run inference
        with torch.no_grad():
            out = self.forward(inputs)

        # Return predictions
        return [Prediction(classification={"real": float(o[0]), "fake": float(o[1])}) for o in out['output']]
