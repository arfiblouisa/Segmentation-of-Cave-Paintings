import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2

def region_boundaries(labels):
    """
    Compute the boundaries of labeled regions in a segmentation mask.

    Parameters
    labels (ndarray): 2D array of integer labels representing segmented regions.

    Returns
    ndarray: Binary image of the same shape as `labels`, where boundary pixels are 255 and others are 0.
    """
    h, w = labels.shape
    boundaries = np.zeros((h, w), dtype=np.uint8)
    for y in range(h - 1):
        for x in range(w - 1):
            if labels[y, x] != labels[y, x + 1] or labels[y, x] != labels[y + 1, x]:
                boundaries[y, x] = 255
    return boundaries



def apply_meanshift(img, spatial_radius, color_radius):
    h, w = img.shape[:2]

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))

    features = np.column_stack((
        xs.flatten(),
        ys.flatten(),
        img.reshape(-1, 3)
    )).astype(np.float32)

    features[:, 0:2] /= spatial_radius
    features[:, 2:5] /= color_radius

    ms = MeanShift(bandwidth=1.0, bin_seeding=True)
    labels = ms.fit_predict(features)
    labels_img = labels.reshape(h, w)

    segmented = np.zeros_like(img)
    for lab in np.unique(labels_img):
        mask = labels_img == lab
        segmented.reshape(-1, img.shape[2])[mask.flatten()] = img.reshape(-1, img.shape[2])[mask.flatten()].mean(axis=0)

    boundaries = region_boundaries(labels_img)
    result = segmented.copy()
    result[boundaries == 255] = 0

    return labels_img, segmented, result


def create_regions_image(img, k_s=80, k_r=24, save_path=None):

    h, w, _ = img.shape
    k_s = k_s * (np.sqrt((h * w) / 136000))

    labels_img, segmented, result = apply_meanshift(
        img,
        spatial_radius=k_s,
        color_radius=k_r
    )

    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_LAB2RGB)

    unique_labels = np.unique(labels_img)

    plt.figure(figsize=(10, 8))
    plt.imshow(segmented_rgb)
    plt.axis("off")

    legend_elements = []

    for lab in unique_labels:
        mask = labels_img == lab

        # Compute mean color of this region
        mean_color = segmented_rgb[mask].mean(axis=0)

        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=f"Region {lab}",
                markerfacecolor=mean_color / 255,
                markersize=8
            )
        )

    plt.legend(handles=legend_elements, loc="upper right")

    if save_path is not None:
        plt.savefig(save_path + '_best_regions.png',
                    bbox_inches='tight',
                    pad_inches=0)

    plt.show()
    plt.close()

    return labels_img, segmented_rgb


def create_segmentation_results(
        regions_to_merge,
        labels_img,
        img,
        save_path=None
):
    # Create binary mask from selected labels
    mask = np.isin(labels_img, list(regions_to_merge))

    original_rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    overlay = original_rgb.copy()

    green = np.array([0, 255, 0], dtype=np.uint8)
    alpha = 0.5

    overlay[mask] = (
        (1 - alpha) * overlay[mask] +
        alpha * green
    ).astype(np.uint8)

    plt.figure(figsize=(10, 8))
    plt.imshow(overlay)
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path + '_segmentation_results.png',
                    bbox_inches='tight',
                    pad_inches=0)

    plt.show()
    plt.close()

    return mask