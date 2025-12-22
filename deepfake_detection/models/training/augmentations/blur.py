import torch
from torchvision.transforms import v2


class RandomGaussianBlur(torch.nn.Module):
    """
    Data augmentation that can be used to apply gaussian blur.
    """

    def __init__(self, sigma_low : float = 0, sigma_up : float = 3, prob: float = 0.1):
        """
        :param sigma_low: Gaussian blur standard deviation lower bound.
        :param sigma_up: Gaussian blur standard deviation upper bound.
        :param prob: Probability of applying Gaussian blur.
        """
        super().__init__()
        self.sigma_low = sigma_low
        self.sigma_up = sigma_up
        self.prob = prob

    def forward(self, img):
        if torch.rand(1) < self.prob:
            kernel_size = int(2 * int(2 * self.sigma + 0.5) + 1)
            return v2.functional.gaussian_blur(img,
                                               [kernel_size, kernel_size],
                                               [self.sigma_low, self.sigma_up])
        return img
