from timm.data.auto_augment import rand_augment_transform
from PIL import Image


class TimmRandAugment:
    """
    PIL-level RandAugment wrapper using timm policy strings.
    """
    def __init__(self, config_str: str = "rand-m9-mstd0.5-inc1", interpolation: str = "bicubic"):
        # timm's PIL ops expect a PIL resampling mode (int/enum), not a raw string.
        if isinstance(interpolation, str):
            key = interpolation.strip().lower()
            mapping = {
                "nearest": Image.Resampling.NEAREST,
                "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC,
            }
            interpolation = mapping.get(key, Image.Resampling.BICUBIC)

        self.transform = rand_augment_transform(config_str, {"interpolation": interpolation})

    def __call__(self, img):
        return self.transform(img)
