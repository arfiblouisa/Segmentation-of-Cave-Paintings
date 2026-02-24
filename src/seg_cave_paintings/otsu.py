import cv2
import numpy as np
import matplotlib.pyplot as plt
import seg_cave_paintings.preprocessing_utils as prep_utils

def create_segmentation_results(mask, img, save_path=None):

    # original_rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    original_rgb = img
    overlay = original_rgb.copy()

    green = np.array([0, 255, 0], dtype=np.uint8)
    alpha = 0.5  # use 0.5 so you actually see transparency

    overlay[mask] = (
        (1 - alpha) * overlay[mask] +
        alpha * green
    ).astype(np.uint8)

    plt.figure(figsize=(10, 8))
    plt.imshow(overlay)
    plt.axis("off")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()


def otsu_mask_from_color_space(rgb):

    # HSV pour rouge 
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    lab = prep_utils.preprocess_image(rgb, color_space="lab")
    L = lab[..., 0]

    # seuil adaptatif noir via Otsu inversé
    _, black_mask = cv2.threshold(
        L, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # contraindre saturation faible pour évite rouge sombre
    low_sat = S < 200
    black_mask = black_mask & low_sat.astype(np.uint8)*255
    mask=black_mask

    return mask