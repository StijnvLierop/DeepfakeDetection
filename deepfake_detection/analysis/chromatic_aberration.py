from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from scipy.optimize import least_squares


def calc_vector_field(img: np.ndarray, x0: int, y0: int, alpha: float, step: int) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    xw = alpha * (x - x0) + x0 - x
    yw = alpha * (y - y0) + y0 - y

    return x, y, xw, yw


def shi_tomasi_corners(image_gray: np.ndarray,
                       max_corners: int=1000,
                       quality_level: float=0.01,
                       min_distance: int=10) -> np.ndarray:
    """
    Computes the Shi-Tomasi corners for a given grayscale image, identifying features such as edges and corners based
    on the Shi-Tomasi algorithm. It selects the strongest corners up to the specified maximum number and determines
    those of sufficient quality and distance from one another.

    :param image_gray: The grayscale image on which to detect corners. It is expected to be a 2D array.
    :param max_corners: The maximum number of strongest corners to return. Defaults to 1000.
    :param quality_level: The minimum quality level of corners to retain, expressed as a fraction of the strongest
                          corner's quality. Defaults to 0.01.
    :param min_distance: The minimum Euclidean distance between two returned corner points. This ensures that
                         returned corners are not too close. Defaults to 10.

    :return: A 2D array of shape (N, 2), where each row contains the (x, y) coordinates of a detected corner.
             The total number of rows, N, is at most `max_corners`.
    """
    corners = cv2.goodFeaturesToTrack(image_gray, max_corners, quality_level, min_distance)
    return corners.reshape(-1, 2).astype(int)


def correlation_coefficient(block1: np.ndarray, block2: np.ndarray) -> float:
    """
    Compute the normalized cross-correlation between two image blocks.

    This function calculates the correlation coefficient between two image blocks, taking into account their
    intensity variations. The input image blocks are converted to float32 for processing. It then centers the
    intensity values by subtracting the mean value of each block and computes the normalized cross-correlation.
    If the computation results in a zero denominator during normalization, the function returns zero.

    :param block1: The first image block of shape (H,W).
    :param block2: The second image block of shape (H,W).
    :return: The normalized cross-correlation coefficient between the two image blocks, ranging from -1 to 1.
    """
    # Compute normalized cross-correlation between two image blocks
    block1 = block1.astype(np.float32)
    block2 = block2.astype(np.float32)
    block1 -= np.mean(block1)
    block2 -= np.mean(block2)
    numerator = np.sum(block1 * block2)
    denominator = np.sqrt(np.sum(block1 ** 2) * np.sum(block2 ** 2))
    if denominator == 0:
        return 0
    return numerator / denominator


def estimate_lateral_chromatic_aberration(img: np.ndarray)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates lateral chromatic aberration of a given image using the method proposed by Mayer et al, 2018.
    The method performs the following steps:
    1. Extract a series of corner points using Shi Tomasi method.
    2. For each corner point, estimate local red/green and blue/green aberration vectors using diamond search.
    3. Estimate global red/green and blue/green aberration parameters from local vectors using least squares.

    @article{Mayer2018,
             title = {Accurate and Efficient Image Forgery Detection Using Lateral Chromatic Aberration},
             volume = {13},
             ISSN = {1556-6021},
             url = {http://dx.doi.org/10.1109/TIFS.2018.2799421},
             DOI = {10.1109/tifs.2018.2799421},
             number = {7},
             journal = {IEEE Transactions on Information Forensics and Security},
             publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
             author = {Mayer,  Owen and Stamm,  Matthew C.},
             year = {2018},
             month = jul,
             pages = {1762–1777}
            }

    :param img: RGB image of shape (H,W).
    :return: The estimated lateral chromatic aberration parameters of the given image.
             - Global red/green displacement parameters (alpha, x0, y0).
             - Global blue/green displacement parameters (alpha, x0, y0).
             - Local red/green displacement vectors for all keypoints of shape (Nx2).
             - Local blue/green displacement vectors for all keypoints of shape (Nx2).
             - Keypoints used for estimation of shape (Nx2).
    """
    # Get keypoints with high gradient
    gray_img = np.array(Image.fromarray(img).convert('L'))
    keypoints = shi_tomasi_corners(gray_img, max_corners=1000, min_distance=30)

    # Estimate local chromatic aberration displacement for each keypoint
    local_displacements_gr, local_displacements_gb = estimate_local_lca_displacements(img,
                                                                                      keypoints,
                                                                                      block_size=15,
                                                                                      search_range=3)

    # Estimate global chromatic aberration displacement from local displacements
    global_displacement_gr = estimate_global_lca_model(keypoints, local_displacements_gr)
    global_displacement_gb = estimate_global_lca_model(keypoints, local_displacements_gb)

    return local_displacements_gr, local_displacements_gb, global_displacement_gr, global_displacement_gb, keypoints


def diamond_search(template: np.ndarray, search_area: np.ndarray) -> Tuple[int, int]:
    """
    Diamond Search algorithm to find the best matching block displacement.

    :param template: 2D numpy array of the reference block (W x W)
    :param search_area: larger 2D numpy array where the template searches for best match.
                        Its size should be (W+2*S) x (W+2*S), where S is the search range.
    :return: displacement as (dx, dy) tuple indicating offset in pixels from center of search_area
    """

    # Define the diamond search patterns relative coordinates
    LDSP = [  # Large diamond search pattern, 8 points around center
        (0, -2), (1, -1), (2, 0), (1, 1),
        (0, 2), (-1, 1), (-2, 0), (-1, -1)
    ]
    SDSP = [  # Small diamond search pattern, 4 points around center
        (0, -1), (1, 0), (0, 1), (-1, 0)
    ]

    # Central point coordinates in search_area
    center_x = search_area.shape[1] // 2
    center_y = search_area.shape[0] // 2
    block_size = template.shape[0]  # Assuming square blocks

    # Evaluate correlation at a given center point offset
    def correlation_at_offset(cx, cy):
        top_left_x = cx - block_size // 2
        top_left_y = cy - block_size // 2
        candidate_block = search_area[top_left_y:top_left_y + block_size, top_left_x:top_left_x + block_size]
        if candidate_block.shape != template.shape:
            return -1  # invalid block (at search area border)
        return correlation_coefficient(template, candidate_block)

    # Start with LDSP
    current_point = (center_x, center_y)
    while True:
        best_corr = -np.inf
        best_point = current_point

        # Search the 8 neighboring points of the LDSP + center point
        candidates = [current_point] + [(current_point[0] + dx, current_point[1] + dy) for dx, dy in LDSP]

        for (cx, cy) in candidates:
            if (0 <= cx - block_size // 2 < search_area.shape[1] - block_size + 1 and
                0 <= cy - block_size // 2 < search_area.shape[0] - block_size + 1):
                corr = correlation_at_offset(cx, cy)
                if corr > best_corr:
                    best_corr = corr
                    best_point = (cx, cy)

        # If best match is at center, switch to SDSP
        if best_point == current_point:
            break
        else:
            current_point = best_point
            # continue LDSP

    # SDSP refinement
    while True:
        best_corr = -np.inf
        best_point = current_point

        candidates = [current_point] + [(current_point[0] + dx, current_point[1] + dy) for dx, dy in SDSP]

        for (cx, cy) in candidates:
            if (0 <= cx - block_size // 2 < search_area.shape[1] - block_size + 1 and
                0 <= cy - block_size // 2 < search_area.shape[0] - block_size + 1):
                corr = correlation_at_offset(cx, cy)
                if corr > best_corr:
                    best_corr = corr
                    best_point = (cx, cy)

        if best_point == current_point:
            break
        else:
            current_point = best_point

    # Return displacement relative to center
    dx = current_point[0] - center_x
    dy = current_point[1] - center_y

    return (dx, dy)


def estimate_local_lca_displacements(image: np.ndarray,
                                     keypoints: np.ndarray,
                                     block_size: int,
                                     search_range: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Local LCA displacement algorithm to find the best matching block displacement for each keypoint.

    :param image: 3D np.ndarray RGB image.
    :param keypoints: 2D numpy array of keypoints (x,y) in the image to find the best local displacements of.
    :param block_size: block size for local displacement.
    :param search_range: search range for local displacement (set larger for larger aberration).
    :return: a tuple of np.ndarrays where first array contains local displacements (dx, dy) for red/green displacement
             and second array contains local displacements (dx, dy) for blue/green displacement for all keypoints.

    """
    # For each keypoint, estimate local LCA displacement between green-red and green-blue channel pairs
    local_displacements_gr = []
    local_displacements_gb = []
    red = image[:, :, 0]  # red channel
    green = image[:, :, 1]  # green channel
    blue = image[:, :, 2]  # blue channel

    # Loop over keypoints
    for (x, y) in keypoints:

        # Extract reference (green) block
        ref_block = green[y - block_size // 2:y + block_size // 2 + 1, x - block_size // 2:x + block_size // 2 + 1]

        # Extract search area in red and blue channels (to find where green area fits best)
        red_search_area = red[y - block_size // 2 - search_range:y + block_size // 2 + search_range + 1,
                          x - block_size // 2 - search_range:x + block_size // 2 + search_range + 1]
        blue_search_area = blue[y - block_size // 2 - search_range:y + block_size // 2 + search_range + 1,
                           x - block_size // 2 - search_range:x + block_size // 2 + search_range + 1]

        # Execute diamond search to find optimal fit
        disp_gr = diamond_search(ref_block, red_search_area)
        disp_gb = diamond_search(ref_block, blue_search_area)

        # Add results to array
        local_displacements_gr.append(disp_gr)
        local_displacements_gb.append(disp_gb)

    return np.array(local_displacements_gr), np.array(local_displacements_gb)


def lca_global_model_residuals(params: np.ndarray, keypoints: np.ndarray, local_d) -> np.ndarray:
    """
    Calculates the residuals of local vectors with a global model paramaterized on the given parameters.

    :param params: The parameters of the global chromatic aberration model (alpha, x0, y0).
    :param keypoints: 2D numpy array of keypoints (x,y) in the image of lenght N.
    :param local_d: 2D numpy array of length N containing local displacements (dx, dy).
    :return: residuals of length N.
    """
    alpha, zx, zy = params
    d_model = alpha * (keypoints - np.array([zx, zy])) + np.array([zx, zy]) - keypoints
    residuals = (local_d - d_model).reshape(-1)
    return residuals


def estimate_global_lca_model(keypoints: np.ndarray, local_displacements: np.ndarray) -> np.ndarray:
    """
    Estimates the global chromatic aberration model parameters from a series of local displacements using least squares.

    :param keypoints: 2D numpy array of keypoints (x,y) in the image of length N.
    :param local_displacements: 2D numpy array of length N containing local displacements (dx, dy).
    :return: best global model parameters (alpha, x0, y0). Alpha indicates the strength of the aberration.
             A larger deviation from 1 results in a stronger aberration effect. x0 and y0 represent the coordinates of
             the optical center op the image.
    """
    # Take no aberration and geometric center of keypoints as intial guess of image center
    initial_guess = [1.0, keypoints[:, 0].mean(), keypoints[:, 1].mean()]

    # Find optimal parameters (alpha, x0, y0)
    optim = least_squares(fun=lca_global_model_residuals, x0=initial_guess, args=(keypoints, local_displacements))

    return optim.x


def plot_keypoints_displacement(keypoints: np.ndarray,
                                local_displacements: np.ndarray,
                                global_params: np.ndarray,
                                title: str,
                                scale: float=0.1,
                                ax=plt.axes) -> None:
    """
    Plots estimated displacement vectors for a given image at each given keypoint.

    :param keypoints: 2D np.ndarray containing the keypoints.
    :param local_displacements: 2D np.ndarray containing the local displacement vectors (x,y) for all keypoints.
    :param global_params: array of parameters (alpha, x0 and y0).
    :param title: Title of the plot.
    :param scale: Scaling factor to scale arrows with.
    :param ax: Axes object to plot on.
    """
    # Plot local displacements
    ax.quiver(keypoints[:, 0], keypoints[:, 1], local_displacements[:, 0], local_displacements[:, 1],
              color='red', label='Local Displacement', scale=scale, scale_units='xy')

    # Plot global diplacements
    alpha, x0, y0 = global_params
    global_v_map = alpha * (keypoints - np.array([x0, y0])) + np.array([x0, y0]) - keypoints
    ax.quiver(keypoints[:, 0], keypoints[:, 1], global_v_map[:, 0], global_v_map[:, 1],
              color='green', label='Global Displacement', scale=scale, scale_units='xy')

    # Plot layout
    ax.set_title(title)
    ax.axis('off')


def plot_vector_field(img: np.ndarray, params: np.ndarray, title: str, scale: float=0.1, ax=plt.axes) -> None:
    """
    Plots a vector field for a given image and set of parameters.

    :param img: 2D or 3D np.ndarray containing the image data.
    :param params: Array of parameters (alpha, x0 and y0).
    :param title: Title of the plot.
    :param scale: Scaling factor to scale arrows with.
    :param ax: Axis to plot on.
    """
    # Get params
    alpha, x0, y0 = params

    # Calculate params for global vector field
    x, y, xw, yw = calc_vector_field(img, x0, y0, alpha, 100)

    # Plot vector field
    ax.quiver(x, y, xw, yw, color='green', scale_units='xy', scale=scale)

    # Plot layout
    ax.set_title(title)
    ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])


def visualize_estimated_chromatic_aberration(img: np.ndarray,
                                             local_rg: np.ndarray,
                                             local_bg: np.ndarray,
                                             global_rg: np.ndarray,
                                             global_bg: np.ndarray,
                                             keypoints: np.ndarray) -> None:
    """
    Plots a figure with the following subplots:
    1. Estimated global aberration vector field for red/green displacement.
    2. Estimated global aberration vector field for blue/green displacement.
    3. Keypoint locations used for estimating local and global aberration parameters.
    4. Global and local red/green displacement vectors for each keypoint.
    5. Global and local blue/green displacement vectors for each keypoint.

    :param img: 2D or 3D np.ndarray containing the image data.
    :param local_rg: 2D np.ndarray containing the local red/green displacement vectors (x,y) for all keypoints.
    :param local_bg: 2D np.ndarray containing the local blue/green displacement vectors (x,y) for all keypoints.
    :param global_rg: array of parameters (alpha, x0 and y0) for red/green global displacement.
    :param global_bg: array of parameters (alpha, x0 and y0) for blue/green global displacement.
    :param keypoints: 2D np.ndarray containing the keypoints.
    """
    # Create figure with several subplots
    fig, axs = plt.subplots(1, 5, figsize=(50, 10))

    # Plot global red/green displacement
    plot_vector_field(img, global_rg,
                      title='Red/Green estimated global displacement (scaled x10)',
                      scale=0.1,
                      ax=axs[0])

    # Plot global blue/green displacement
    plot_vector_field(img, global_bg,
                      title='Blue/Green estimated global displacement (scaled x10)',
                      scale=0.1,
                      ax=axs[1])

    # Plot keypoints
    axs[2].imshow(img)
    axs[2].scatter(keypoints[:, 0], keypoints[:, 1], color='blue')
    axs[2].set_title('Selected Keypoints for Local Estimates')
    axs[2].axis('off')

    # Plot estimated red/green global and local displacement for keypoints
    plot_keypoints_displacement(img,
                                keypoints,
                                local_rg,
                                global_rg,
                                title='Red/Green displacement (scaled x10)',
                                ax=axs[3])

    # Plot estimated blue/green global and local displacement for keypoints
    plot_keypoints_displacement(img,
                                keypoints,
                                local_bg,
                                global_bg,
                                title='Blue/Green displacement (scaled x10)',
                                ax=axs[4])

    # Show figure
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 0.5))
    plt.axis('equal')
    plt.show()


def simulate_lateral_chromatic_aberration(image: np.ndarray, alpha: float, center: Tuple[int, int]=None) -> np.ndarray:
    """
    Simulates lateral chromatic aberration by radially shifting red and blue channels outward.

    :param image: Input RGB image as np.ndarray (H x W x 3).
    :param alpha: quantifies how much the chromatic aberration displacement increases per pixel of distance from
                  the optical center. Larger deviation from 1 results in a stronger aberration effect.
    :param center: (x, y) center for radial shift. If None, uses image center.
    :return: Output RGB image with simulated chromatic aberration.
    """
    # Determine center of aberration
    h, w = image.shape[:2]
    y_idx, x_idx = np.indices((h, w))
    if center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = center

    # r: pixel coordinates; zeta: optical center
    r = np.stack([x_idx, y_idx], axis=-1)
    zeta = np.array([cx, cy])

    # According to the paper's model:
    # d(r, theta) = alpha * (r - zeta) + zeta - r = (alpha - 1) * (r - zeta)
    delta = (r - zeta)
    disp = (alpha - 1.0) * delta

    # Red channel: outward shift (by +disp)
    map_x_r = (x_idx + disp[..., 0]).astype(np.float32)
    map_y_r = (y_idx + disp[..., 1]).astype(np.float32)

    # Blue channel: inward shift (by -disp)
    map_x_b = (x_idx - disp[..., 0]).astype(np.float32)
    map_y_b = (y_idx - disp[..., 1]).astype(np.float32)

    # Warp red and blue channels
    red, green, blue = cv2.split(image)
    red_shifted = cv2.remap(red, map_x_r, map_y_r, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    blue_shifted = cv2.remap(blue, map_x_b, map_y_b, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Combine warped channels with unchanged green channel
    result = cv2.merge([red_shifted, green, blue_shifted])

    return result


def inconsistency(local_d, global_params, keypoints) -> np.ndarray:
    """
    Calculate inconsistency between local displacement and global displacement.

    :param local_d: 2D np.ndarray containing local displacement vectors (x,y) of length N.
    :param global_params: array of parameters (alpha, x0 and y0) of the global displacement model.
    :param keypoints: 2D np.ndarray containing the keypoints of length N.
    """
    alpha, x0, y0 = global_params
    d_model = alpha * (keypoints - np.array([x0, y0])) + np.array([x0, y0]) - keypoints
    return (local_d - d_model) / alpha
