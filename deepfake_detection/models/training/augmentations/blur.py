from typing import Tuple

import torch
from torchvision.transforms import v2


class RandomGaussianBlur(torch.nn.Module):
    """
    Data augmentation that can be used to apply gaussian blur.
    """

    def __init__(self, sigma: Tuple[float, float] = (0, 3), prob: float = 0.1):
        """
        :param sigma: Gaussian blur standard deviation lower and upper bound.
        :param prob: Probability of applying Gaussian blur.
        """
        super().__init__()
        self.sigma = sigma
        self.prob = prob

    def forward(self, img):
        if torch.rand(1) < self.prob:
            kernel_size = int(2 * int(2 * self.sigma[0] + 0.5) + 1)
            return v2.functional.gaussian_blur(
                img,
                [kernel_size, kernel_size],
                [self.sigma[0] + 0.00000000001, self.sigma[1]],
            )
        return img
