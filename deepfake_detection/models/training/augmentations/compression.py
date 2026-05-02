import io
from typing import Tuple, Sequence

import torch
from PIL import Image
from torchvision.io import encode_jpeg, decode_jpeg


class RandomPILJpegCompression(torch.nn.Module):
    """
    Data augmentation that can apply random JPEG compression via PIL or OpenCV.
    """

    def __init__(
        self,
        quality_range: Tuple[float, float] = (30, 100),
        prob: float = 0.5,
        methods: Sequence[str] = ("pil",),
    ):
        """
        :param quality_range: Range of JPEG compression qualities to choose from.
        :param prob: Probability of applying random JPEG compression.
        :param methods: JPEG backends to sample from (e.g., ("pil", "cv2")).
        """
        super().__init__()
        self.quality_range = quality_range
        self.prob = prob
        self.methods = tuple(methods)

    def forward(self, img):
        if torch.rand(1).item() > self.prob:
            return img

        low, high = self.quality_range
        quality = int(torch.randint(low, high + 1, (1,)).item())
        method = self.methods[torch.randint(0, len(self.methods), (1,)).item()]

        if isinstance(img, torch.Tensor):
            # Tensor path: encode_jpeg requires uint8 [C, H, W]
            compressed_bytes = encode_jpeg(img, quality=quality)
            return decode_jpeg(compressed_bytes)

        if method == "cv2":
            try:
                import cv2
                import numpy as np
            except ImportError:
                # Fall back to PIL when OpenCV is not installed.
                method = "pil"
            else:
                img_arr = np.array(img.convert("RGB"))
                img_bgr = img_arr[:, :, ::-1]
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                success, enc = cv2.imencode(".jpg", img_bgr, encode_param)
                if success:
                    dec = cv2.imdecode(enc, 1)
                    return Image.fromarray(dec[:, :, ::-1])

        # PIL path
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()
