import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
from PIL import Image, ImageFilter


"""
Techniques for chromatic aberration estimation are based on:
@inproceedings{Johnson2006,
  series = {MM&Sec ’06},
  title = {Exposing digital forgeries through chromatic aberration},
  url = {http://dx.doi.org/10.1145/1161366.1161376},
  DOI = {10.1145/1161366.1161376},
  booktitle = {Proceedings of the 8th workshop on Multimedia and security},
  publisher = {ACM},
  author = {Johnson,  Micah K. and Farid,  Hany},
  year = {2006},
  month = sep,
  pages = {48–55},
  collection = {MM&Sec ’06}
}
"""


def warp_channel(channel: np.ndarray, x0: int, y0: int, alpha: float) -> np.ndarray:
    """
    Warp channel image of the form (H x W) by expanding/contracting around center coordinates (x0, y0) with scaling alpha.

    :param channel: Channel image (2D array of the form (H x W)).
    :param x0: Center x coordinate.
    :param y0: Center y coordinate.
    :param alpha: Scaling factor.
    :return: The warped channel image (same shape as input).
    """
    # Create sampling lattice for warped image with same dimensions as unwarped image
    h, w = channel.shape

    # Calculate warped coordinates given center of image and scaling factor
    x, y, xw, yw = calc_vector_field(channel, x0, y0, alpha, step=1)

    # Create warped image from original coordinates to warped coordinates by interpolation
    coords = np.array([xw.ravel(), yw.ravel()])
    warped = map_coordinates(channel, coords, order=1, mode='reflect').reshape(h, w)

    return warped


def compute_mutual_information(im1: np.ndarray, im2: np.ndarray, bins=256) -> float:
    """
    Computes the mutual information between two images.

    :param im1: First image of shape (H x W).
    :param im2: Second image of shape (H x W).
    :param bins: Number of bins for histogram computation. More bins means a more accurate estimate.
    :return: The mutual information between the two images.
    """
    # Use numpy's histogramdd which is faster than histogram2d for 2D data
    hgram = np.histogramdd([im1.ravel(), im2.ravel()], bins=bins)[0]
    return mutual_info_score(None, None, contingency=hgram)


def objective(params, channel, green):
    """
    Objective function for optimization. Minimizes mutual information between two images (channel and green).

    :param params: (x0, y0, alpha).
    :param channel: Red or Blue channel of the form (height, width), 2D array.
    :param green: Green channel, 2D array of the form (height, width).
    """
    x0, y0, alpha = params
    cw = warp_channel(channel, x0, y0, alpha)
    mi = compute_mutual_information(cw, green)
    return -mi  # Negative because we want to maximize MI


def optimize_parameters(channel, green, search_center, search_alpha, num_center_steps=5, num_alpha_steps=5):
    """
    Brute force search for (x0, y0, alpha) maximizing mutual information between two images (channel and green).

    :param channel: Red or Blue channel of the form (height, width), 2D array.
    :param green: Green channel, 2D array.
    :param search_center: (min_x, max_x, min_y, max_y).
    :param search_alpha: (min_alpha, max_alpha).
    :return: (best_x0, best_y0, best_alpha), warped_channel.
    """
    # Define parameter grids
    min_x, max_x, min_y, max_y = search_center
    min_alpha, max_alpha = search_alpha
    x_vals = np.linspace(min_x, max_x, num_center_steps)
    y_vals = np.linspace(min_y, max_y, num_center_steps)
    alpha_vals = np.linspace(min_alpha, max_alpha, num_alpha_steps)

    best_mi = float('-inf')
    best_params = None

    # Brute force search over parameter grid
    for x0 in x_vals:
        for y0 in y_vals:
            for alpha in alpha_vals:
                params = [x0, y0, alpha]
                curr_mi = -objective(params, channel, green)
                if curr_mi > best_mi:
                    best_mi = curr_mi
                    best_params = params

    return best_params


def estimate_lateral_chromatic_aberration_parameters(img: np.ndarray):
    """
    This function estimates the parameters for lateral chromatic aberration (red-green and blue-green mismatch).

    :param img: Image (3D array of the form (height, width, channels)).
    :return: (params_r, params_b), where params_r and params_b are the estimated parameters
             for red-green and blue-green mismatch, respectively.
    """

    # Split image into channels (assuming RGB image)
    red = img[:, :, 0].astype(np.float32)
    green = img[:, :, 1].astype(np.float32)
    blue = img[:, :, 2].astype(np.float32)

    # Define search space for parameter search
    h, w = green.shape
    search_center = (w // 4, w // 4 * 3, h // 4, h // 4 * 3)  # restrict search to the center half of image
    search_alpha = (0.9987, 1.009)

    # Estimate parameters for red-green aberration
    params_r = optimize_parameters(red, green, search_center, search_alpha)

    # Estimate parameters for blue-green mismatch
    params_b = optimize_parameters(blue, green, search_center, search_alpha)

    return params_r, params_b


def calc_vector_field(img: np.ndarray, x0: int, y0: int, alpha: float, step: int):
    """
    Calculates a vector field of the chromatic aberration warp for a given image (channel).

    :param img: Image (2D array).
    :param x0: Center x coordinate.
    :param y0: Center y coordinate.
    :param alpha: Scaling factor.
    :param step: Step size for grid.
    :return vector_field: Vector field of shape (x, y, xw, yw). Here, x and y are the original coordinates
                          and xw and yw are the warped coordinates.
    """

    # Set up grid for vector field
    y, x = np.mgrid[0:img.shape[0]:step, 0:img.shape[1]:step]
    xw = (alpha - 1) * (x - x0)
    yw = (alpha - 1) * (y - y0)

    return x, y, xw, yw


def plot_chromatic_aberration_displacement(img,
                                           params,
                                           step=32,
                                           scale=50
                                           ):
    """
    This function plots the displacement of the chromatic aberration warp for a given image (channel).

    :param img: Image (2D array) of the form (height, width).
    :param params: (x0, y0, alpha).
    :param step: Step size for grid.
    :param scale: Scale factor for vector field.
    """
    # Get params
    x0, y0, alpha = params

    # Calculate vector field
    x, y, u, v = calc_vector_field(img, x0, y0, alpha, step=step)

    plt.figure(figsize=(10, 10))

    # Show vector field (quiver plot)
    plt.imshow(img, cmap='grey')
    plt.quiver(
        x, y, u*scale, v*scale, angles='xy', scale_units='xy', scale=1, color='red', width=0.003, headwidth=2
    )

    # Plot estimated center
    plt.plot(x0, y0, 'bo', label='Estimated center')

    # Plot layout
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_patch_errors(img, errors, window_size, cmap='jet', title='Patch Angular Errors Overlay'):
    """
    Plots an overlay of patch angular errors on a blank image canvas.

    :param img: np.ndarray (height, width) of the image canvas.
    :param errors: list of tuples (x, y, angular_error), where (x,y) is the top-left corner of the patch.
    :param window_size: int, size of the square patch/window.
    :param cmap: string, matplotlib colormap name to map error magnitude to colors.
    :param title: string, title for the plot.
    """
    H, W, C = img.shape

    # Arrays to accumulate error sums and counts per pixel for averaging overlaps
    error_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    # Fill error_map by adding angular errors in the patch areas
    for (x, y, error) in errors:
        x_end = min(x + window_size, W)
        y_end = min(y + window_size, H)
        error_map[y:y_end, x:x_end] += error
        count_map[y:y_end, x:x_end] += 1

    # Avoid division by zero and compute average angular errors per pixel
    count_map[count_map == 0] = 1
    avg_error_map = error_map / count_map

    # Plot the heatmap of angular errors over the image area
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='gray')
    im = plt.imshow(avg_error_map, cmap=cmap, alpha=0.5)
    plt.colorbar(im, label='Angular Error (degrees)')
    plt.title(title)
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.show()


def angular_error(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the angular error between two vectors.

    :param v1: First vector.
    :param v2: Second vector.
    :return: The angular error between the two vectors.
    """
    dot_product = np.dot(v1, v2.T)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-15)
    angle = np.arccos(cos_theta)
    return angle


def get_image_patches_with_largest_gradient(img, patch_size, top_k=50):
    """
    Extract patches with the largest gradient magnitude from an image.

    :param image_path: Path to the input image.
    :param patch_size: Size of each patch (width, height).
    :param top_k: Number of top patches to return with largest gradient.
    :return: list of tuple: List of top_k patches as (x, y, score), where (x, y) is the top-left corner coordinate.
    """
    # Load image and convert to grayscale
    gray_img = Image.fromarray(img).convert('L')

    # Compute horizontal and vertical gradients using Sobel-like kernels
    grad_x = gray_img.filter(ImageFilter.Kernel((3,3), [-1,0,1,-2,0,2,-1,0,1], scale=1))
    grad_y = gray_img.filter(ImageFilter.Kernel((3,3), [-1,-2,-1,0,0,0,1,2,1], scale=1))

    grad_x_arr = np.array(grad_x, dtype=np.float32)
    grad_y_arr = np.array(grad_y, dtype=np.float32)

    # Gradient magnitude
    grad_mag = np.sqrt(grad_x_arr**2 + grad_y_arr**2)

    width, height = gray_img.size
    patch_w, patch_h = patch_size

    patches_coords = []
    # Slide over image in patch-sized windows
    for y in range(0, height - patch_h + 1, patch_h):
        for x in range(0, width - patch_w + 1, patch_w):
            patch_grad = grad_mag[y:y+patch_h, x:x+patch_w]
            score = np.sum(patch_grad)
            patches_coords.append((x, y, score))

    # Sort patches by gradient score in descending order
    patches_coords.sort(key=lambda x: x[2], reverse=True)

    # Get patches
    patches = np.array([img[y:y+patch_h, x:x+patch_w] for x, y, _ in patches_coords[:top_k]])

    # Return top_k patches with coordinates and scores
    return patches_coords[:top_k], patches


def detect_tampering_blocks(image: np.ndarray,
                            window_size: int,
                            global_r_params: np.ndarray,
                            global_b_params: np.ndarray,
                            top_k: int = 50):
    """
    This function detects tampering blocks in an image.

    :param image: Image (3D array of the form (height, width, channels)).
    :param window_size: Size of the square patch/window.
    :param global_r_params: Global parameters for red-green aberration.
    :param global_b_params: Global parameters for blue-green mismatch.
    :param top_k: Number of top patches to return with largest gradient.
    """

    # Get patches with the largest gradient
    coords, patches = get_image_patches_with_largest_gradient(image, (window_size, window_size), top_k=top_k)

    # Loop over blocks
    errors = []
    for coords, patch in tqdm(zip(coords, patches), total=len(patches)):

        # Split
        x, y = coords[:2]

        # Estimate parameters for window
        local_r_params, local_b_params = estimate_lateral_chromatic_aberration_parameters(patch)

        # Global vector fields from window full channels with global params
        xg_rg, yg_rg, xwg_rg, ywg_rg = calc_vector_field(patch[:, :, 0], *global_r_params, 1)
        xg_bg, yg_bg, xwg_bg, ywg_bg = calc_vector_field(patch[:, :, 2], *global_b_params, 1)

        # Local vector fields for current block and local params
        _, _, xwl_rg, ywl_rg = calc_vector_field(patch[:, :, 0], *local_r_params, 1)
        _, _, xwl_bg, ywl_bg = calc_vector_field(patch[:, :, 2], *local_b_params, 1)

        vec_global_rg = np.stack([xwg_rg.flatten(), ywg_rg.flatten()])
        vec_local_rg = np.stack([xwl_rg.flatten(), ywl_rg.flatten()])
        vec_global_bg = np.stack([xwg_bg.flatten(), ywg_bg.flatten()])
        vec_local_bg = np.stack([xwl_bg.flatten(), ywl_bg.flatten()])

        ang_err_rg = angular_error(vec_global_rg, vec_local_rg)
        ang_err_bg = angular_error(vec_global_bg, vec_local_bg)

        mean_ang_err_rg = np.mean(ang_err_rg)
        mean_ang_err_bg = np.mean(ang_err_bg)

        errors.append([x, y, mean_ang_err_rg, mean_ang_err_bg])

    return errors

