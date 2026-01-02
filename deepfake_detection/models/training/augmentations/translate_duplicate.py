import math

from PIL import Image
import torch


class TranslateDuplicate(torch.nn.Module):
    """
    Translates and duplicates the image to a given size.

    Function from https://github.com/chuangchuangtan/NPR-DeepfakeDetection.
    """
    def __init__(self, cropSize: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cropSize = cropSize

    def forward(self, img):
        if min(img.size) < self.cropSize:
            width, height = img.size

            new_width = width * math.ceil(self.cropSize / width)
            new_height = height * math.ceil(self.cropSize / height)

            new_img = Image.new('RGB', (new_width, new_height))
            for i in range(0, new_width, width):
                for j in range(0, new_height, height):
                    new_img.paste(img, (i, j))
            return new_img
        else:
            return img