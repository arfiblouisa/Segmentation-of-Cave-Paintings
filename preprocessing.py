import cv2
import numpy as np


# Color space conversion
def convert_color_space(img, color_space):
    if color_space == "rgb":
        return img

    if color_space == "gray":
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if color_space == "hsv":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if color_space == "lab":
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    raise ValueError(f"Unsupported color_space: {color_space}")


# Histogram equalization
def equalize_image(img):
    if img.ndim == 2:
        return cv2.equalizeHist(img)

    # per-channel equalization
    channels = cv2.split(img)
    eq_channels = [cv2.equalizeHist(c) for c in channels]
    return cv2.merge(eq_channels)


# CLAHE (adaptive equalization)
def clahe_image(img, clip_limit=2.0, tile_grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    if img.ndim == 2:
        return clahe.apply(img)

    channels = cv2.split(img)
    out = [clahe.apply(c) for c in channels]
    return cv2.merge(out)


# Decorrelation stretching
def decorrelation_stretch(img):
    """
    Implements decorrelation stretching using PCA whitening.
    Works on color images only.
    """
    if img.ndim != 3:
        return img

    img = img.astype(np.float32)
    h, w, c = img.shape

    flat = img.reshape(-1, c)

    mean = np.mean(flat, axis=0)
    centered = flat - mean

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # whitening transform
    D = np.diag(1.0 / np.sqrt(eigvals + 1e-5))
    whiten = eigvecs @ D @ eigvecs.T

    stretched = centered @ whiten.T

    # normalize back to 0–255
    stretched -= stretched.min()
    stretched /= stretched.max()
    stretched *= 255

    return stretched.reshape(h, w, c).astype(np.uint8)


# Noise reduction
def denoise_image(img, method=None, gaussian_sigma=1.0, median_ksize=5):
    if method is None:
        return img

    if method == "gaussian":
        k = int(gaussian_sigma * 6 + 1) | 1
        return cv2.GaussianBlur(img, (k, k), gaussian_sigma)

    if method == "median":
        if median_ksize % 2 == 0:
            median_ksize += 1
        return cv2.medianBlur(img, median_ksize)

    raise ValueError(f"Unknown denoise method: {method}")


# Main preprocessing function
def preprocess_cave_image(
    img,
    color_space="rgb",
    contrast=None,               # None | "equalize" | "clahe"
    noise_filter=None,           # None | "gaussian" | "median"
    gaussian_sigma=1.0,
    median_ksize=5,
    decorrelation=False,
    clahe_clip=2.0,
    clahe_tile=(8, 8),
):
    """
    Preprocess cave painting image.

    Parameters
    ----------
    img : np.ndarray (RGB)
    color_space : str
    contrast : str
    noise_filter : str
    decorrelation : bool

    Returns
    -------
    np.ndarray
    """

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # color conversion
    img = convert_color_space(img, color_space)

    # decorrelation stretch
    # if decorrelation and img.ndim == 3:
    #     img = decorrelation_stretch(img)

    # contrast enhancement
    if contrast == "equalize":
        img = equalize_image(img)

    elif contrast == "clahe":
        img = clahe_image(img, clahe_clip, clahe_tile)

    # denoising 
    img = denoise_image(img, noise_filter, gaussian_sigma, median_ksize)

    return img

if __name__=='__main__':
    # img = cv2.imread("dataset\chauvet\panel-of-the-lions-detail-chauvet-cave-6345.jpg")
    img = cv2.imread("dataset/lascaux/Lascaux_027.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    window_name = 'before'

    # Using cv2.imshow() method
    # Displaying the image
    cv2.imshow(window_name, img)
    out = preprocess_cave_image(
        img,
        color_space="rgb",
        # contrast="clahe",
        contrast="equalize",
        # noise_filter="median",
        noise_filter=None,
        median_ksize=5,
        decorrelation=True
    )
    
    window_name = 'Panel of the lions'

    # Using cv2.imshow() method
    # Displaying the image
    cv2.imshow(window_name, out)

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
