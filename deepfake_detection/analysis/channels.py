from typing import Sequence

import numpy as np


def channel_threshold_map(
    img: np.ndarray,
    channels: Sequence[int],
    threshold: float,
    mode: str = "any",
) -> np.ndarray:
    """
    Returns a copy of the image where only pixels that satisfy the threshold condition on the
    specified channels are kept; all other pixels are zeroed out.

    :param img: A numpy array of shape (height, width, channels).
    :param channels: Indices of the channels to evaluate (e.g. [0] for red, [0, 2] for red and blue).
    :param threshold: Pixel values strictly above this value pass the condition.
    :param mode: How to combine the per-channel conditions across the selected channels.
                 - 'any': a pixel passes if *any* selected channel exceeds the threshold.
                 - 'all': a pixel passes if *all* selected channels exceed the threshold.
    :return: A numpy array of the same shape as ``img`` with non-passing pixels set to zero.
    """
    if img.ndim != 3:
        raise ValueError(
            "Image must have shape (height, width, channels). "
            "Got ndim={}.".format(img.ndim)
        )
    if mode not in ("any", "all"):
        raise ValueError("mode must be 'any' or 'all', got '{}'.".format(mode))
    if not channels:
        raise ValueError("channels must contain at least one channel index.")

    channel_masks = np.stack(
        [img[:, :, c] > threshold for c in channels], axis=-1
    )

    if mode == "any":
        mask = channel_masks.any(axis=-1)
    else:
        mask = channel_masks.all(axis=-1)

    result = img.copy()
    result[~mask] = 0
    return result
