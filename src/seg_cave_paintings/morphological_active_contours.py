import numpy as np
import cv2
from skimage.util import img_as_float
from skimage.segmentation import morphological_geodesic_active_contour
from skimage.segmentation import inverse_gaussian_gradient
import matplotlib.pyplot as plt


def store_evolution_in(lst):
    """
    Create a callback function to store intermediate level sets during contour evolution.

    Parameters
    lst (list): List where each intermediate level set will be appended.

    Returns
    function: Callback function to pass to the active contour algorithm.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


def apply_active_contours(img, num_iter=500, threshold=0.75):

    """
    Apply Morphological Geodesic Active Contour (MGAC) to segment an image.

    Parameters
    img (ndarray): Input image, expected to have at least one channel.
    num_iter (int, optional): Number of iterations for contour evolution. Default is 500.
    threshold (float, optional): Convergence threshold for stopping evolution. Default is 0.75.

    Returns
    snake_mask (ndarray): Final binary mask of the segmented region.
    evolution (list of ndarray): List of intermediate level sets during evolution, useful for visualization.
    """

    img=img[:,:,0] # Use only the L channel
    img = img_as_float(img)

    gimage = inverse_gaussian_gradient(img)

    init_ls = np.zeros(img.shape[:2], dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1 # Initialize level set slightly inside the borders

    evolution = []
    callback = store_evolution_in(evolution)
    snake_mask = morphological_geodesic_active_contour(
        gimage,
        num_iter=num_iter,
        init_level_set=init_ls,
        smoothing=1,
        balloon=-1,
        threshold=threshold,
        iter_callback=callback,
    )

    return snake_mask, evolution


def show_contour(img, snake_mask, save_path=None):
    """
    Display an image with an overlaid active contour.

    Parameters
    img (ndarray): Input image, expected to have at least one channel.
    snake_mask (ndarray): Binary mask representing the final contour.
    save_path (str, optional): Path prefix to save the figure. If None, the figure is not saved.

    Returns
    None
    """
    plt.figure()
    plt.imshow(img[:,:,0], cmap='gray') # Show the L channel
    plt.contour(snake_mask, [0.5], colors='r')
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path + '_filtered_active_contours.png',
                    bbox_inches='tight',
                    pad_inches=0,
                    dpi=300)
    plt.show()


def create_segmentation_results(snake_mask, img, save_path=None, alpha=0.5):
    """
    Visualize segmentation results by overlaying the contour mask on the image.
    Converts the input image from LAB to RGB, applies a
    semi-transparent green overlay where the binary snake mask is True,
    and displays the result.

    Parameters
    snake_mask (ndarray): Binary mask representing the segmented region.
    img (ndarray): Input image in LAB color space.
    save_path (str, optional): Path prefix to save the figure. If None, the figure is not saved.
    alpha (float, optional): Transparency factor for the overlay, between 0 (transparent) and 1 (opaque). Default is 0.5.

    Returns
    None
    """
    mask = snake_mask.astype(bool)

    original_rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    overlay = original_rgb.copy()

    green = np.array([0, 255, 0], dtype=np.uint8)

    overlay[mask] = ((1 - alpha) * overlay[mask] + alpha * green).astype(np.uint8) # Apply green mask with transparency inside the segmented region

    plt.figure(figsize=(10, 8))
    plt.imshow(overlay)
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path + '_segmentation_results.png',
                    bbox_inches='tight',
                    pad_inches=0,
                    dpi=300)
    plt.show()
    plt.close()