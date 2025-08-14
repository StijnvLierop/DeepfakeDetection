import os

import numpy as np
import scipy
import torch
from PIL import ImageFilter, Image
import itertools

from deepfake_detection.analysis.dncnn.network_dncnn import DnCNN


def noise_residual(img: np.array, image_filter: str='median') -> np.ndarray:
    """
    This function calculates the noise residual of a given image.

    :param img: A numpy array containing the image data to be transformed.
    :param image_filter: The filter to use for denoising the image. Must be one of:
                         - 'median': applies a Median filter.
                         - 'laplace': applies a Laplace filter.
                         - 'dncnn': applies the DnCNN denoising method proposed by Zhang et al. (2017).
    :return: A numpy array containing the noise residual.
    """
    # Apply filter to get denoised image
    if image_filter == 'median':
        denoised_img = np.array(Image.fromarray(img).filter(ImageFilter.MedianFilter()), dtype=np.float32)
    elif image_filter == 'laplace':
        denoised_img = scipy.ndimage.filters.laplace(np.array(img, dtype=np.float32))
    elif image_filter == 'dncnn':
        denoised_img = denoise_dncnn(img)
    else:
        raise ValueError("Invalid filter. Must be one of: 'median', 'laplace' or 'dncnn'.")

    # Calculate noise residual
    residual = denoised_img - img

    return residual


def denoise_dncnn(img: np.ndarray) -> np.ndarray:
    """
    Implements the denoising method proposed by Zhang et al. (2017).

    Uses the implementation from https://github.com/cszn/DnCNN.git

    @article{zhang2017beyond,
             title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
             author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
             journal={IEEE Transactions on Image Processing},
             year={2017},
             volume={26},
             number={7},
             pages={3142-3155},
            }

    :param img: A numpy array containing the image data to be denoised of
                shape (height, width, channels) or (height, width).
    :return: A numpy array containing the denoised image of the same shape as the input image.
    """
    # Set nr of image channels
    n_channels = img.shape[2] if img.ndim == 3 else 1

    # Define model based on n_channels
    current_dir = os.path.dirname(__file__)
    if n_channels == 1:
        model_path = os.path.join(current_dir, 'dncnn/dncnn_gray_blind.pth')
        img = np.expand_dims(img, axis=2)
    else:
        model_path = os.path.join(current_dir, 'dncnn/dncnn_color_blind.pth')

    # Define and load model
    model = DnCNN(in_nc=n_channels, out_nc=n_channels, nc=64, nb=20, act_mode='R')
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Prepare tensor
    img_tensor = torch.from_numpy(img).float().permute((2, 0, 1)).unsqueeze(0).to(device) / 255

    # Denoise image
    denoised_img = model(img_tensor).squeeze(0).float().clamp_(0, 1).permute((1, 2, 0)).cpu().numpy() * 255
    denoised_img = denoised_img.astype(np.uint8)

    # If single channel image was provided, remove extra channel
    if n_channels == 1:
        denoised_img = denoised_img[:, :, 0]

    return denoised_img


def channel_noise_imbalance_ratio(img: np.ndarray,
                                  image_filter: str='median') -> float:
    """
    This function calculates the channel noise imbalance ratio (CNIR) for a given image.
    The CNIR quantifies the balance between the noise in different image channels. This balance might deviate
    from real images for certain generative models and therefore could be a useful feature.

    :param img: An numpy array containing the image data of shape (height, width, channels).
    :param image_filter: The filter to use for denoising the image. Must be one of:
                         - 'median': applies a Median filter.
                         - 'laplace': applies a Laplace filter.
    :return: The CNIR value for the given image.
    """
    # Ensure the image has a channel dimension
    if img.ndim != 3:
        raise ValueError("Image must have a channel dimension. "
                         "Please ensure the provide image has shape (height, width, channels).")

    # Calculate noise residual for each channel
    residuals = []
    for c in range(img.shape[2]):
        residuals.append(noise_residual(img[:, :, c], image_filter=image_filter))

    # Calculate difference of all channel combinations
    pairs = itertools.combinations(residuals, 2)
    diffs = []
    for p1, p2 in pairs:
        diffs.append(np.mean(np.abs(p1.astype(float) - p2.astype(float))))

    # Calculate mean and standard deviation of noise
    mu_noise = np.mean(np.array(diffs))
    imbalance = np.std(np.array(diffs))

    # Calculate CNIR by dividing the standard deviation of the noise between all channel combinations
    # by the mean noise of all channels
    cnir = imbalance / mu_noise

    return cnir

