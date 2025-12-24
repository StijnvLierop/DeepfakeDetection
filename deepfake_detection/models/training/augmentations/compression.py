from typing import Tuple

import torch
from torchvision.io import encode_jpeg, decode_jpeg


class RandomPILJpegCompression(torch.nn.Module):
    """
    Data augmentation that can be used to apply random JPEG compression using PIL.
    """

    def __init__(self, quality_range: Tuple[float, float] = (30, 100), prob: float = 0.5):
        """
        :param quality_range: Range of JPEG compression qualities to choose from.
        :param prob: Probability of applying random JPEG compression.
        """
        super().__init__()
        self.quality_range = quality_range
        self.prob = prob

    def forward(self, img):
        # img must be a Tensor of type uint8 (0-255)
        if torch.rand(1).item() > self.prob:
            return img

        # Pick a random quality
        low, high = self.quality_range
        quality = int(torch.randint(low, high + 1, (1,)).item())

        # Encode to JPEG (returns a 1D uint8 Tensor of compressed bytes)
        # Note: encode_jpeg expects [C, H, W]
        compressed_bytes = encode_jpeg(img, quality=quality)

        # Decode back to [C, H, W] Tensor
        return decode_jpeg(compressed_bytes)
