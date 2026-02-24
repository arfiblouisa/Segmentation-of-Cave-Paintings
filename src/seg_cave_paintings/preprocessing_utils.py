import cv2

def convert_color_space(img, color_space):
    """
    Convert an RGB image to a specified color space.

    Parameters
    img (ndarray): Input RGB image.
    color_space (str): Target color space: "rgb", "gray", "hsv", or "lab".

    Returns
    ndarray: Image converted to the requested color space.
    """
    if color_space == "rgb":
        return img  # No conversion needed
    if color_space == "gray":
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    if color_space == "hsv":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)   # Convert to HSV
    if color_space == "lab":
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)   # Convert to LAB
    raise ValueError(f"Unsupported color space: {color_space}")




def equalize_image(img):
    """
    Apply histogram equalization to enhance contrast.

    For grayscale images, standard equalization is applied.
    For color images, equalization is applied independently on each channel.

    Parameters
    img (ndarray): Input image (grayscale or color).

    Returns
    ndarray: Contrast-enhanced image.
    """
    if img.ndim == 2:
        return cv2.equalizeHist(img)  # Grayscale image
    ch = cv2.split(img)               # Split color channels
    ch = [cv2.equalizeHist(c) for c in ch]  # Equalize each channel separately
    return cv2.merge(ch)              # Merge channels back



def clahe_image(img, clip=2.0, grid=(8,8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Parameters
    img (ndarray): Input image (grayscale or color).
    clip (float, optional): Threshold for contrast clipping.
    grid (tuple of int, optional): Size of the grid for adaptive histogram equalization.

    Returns
    ndarray:CLAHE-enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    if img.ndim == 2:
        return clahe.apply(img)     # Grayscale
    ch = cv2.split(img)
    ch = [clahe.apply(c) for c in ch]  # Apply CLAHE to each channel
    return cv2.merge(ch)


def denoise(img, method=None, g_sigma=1.0, m_ksize=5):
    """
    Apply denoising to an image using Gaussian or median filtering.

    Parameters
    img (ndarray): Input image.
    method (str, optional): Denoising method. Options are "gaussian", "median", or None (no denoising). Default is None.
    g_sigma (float, optional): Standard deviation for Gaussian blur. Ignored if method is not "gaussian". Default is 1.0.
    m_ksize (int, optional): Kernel size for median blur; must be odd. Ignored if method is not "median". Default is 5.

    Returns
    ndarray: Denoised image.
    """
    if method is None:
        return img
    if method == "gaussian":
        k = int(g_sigma*6+1)|1
        return cv2.GaussianBlur(img,(k,k),g_sigma)
    if method == "median":
        if m_ksize % 2 == 0:
            m_ksize += 1
        return cv2.medianBlur(img, m_ksize)
    raise ValueError(method)


def preprocess_image(img, color_space="lab"):
    """
    Preprocess a RGB image for segmentation.

    Steps:
    1. Convert the image to LAB color space for better perceptual uniformity.
    2. Apply CLAHE to enhance local contrast and reveal details.
    3. Denoise

    Parameters
    img (ndarray): Input RGB image.

    Returns
    ndarray: Preprocessed image suitable for subsequent segmentation.
    """
    out = convert_color_space(img, color_space)
    out = clahe_image(out)
    out = denoise(out, "median", m_ksize=2)
    return out


def crop_image(img_path, x1,y1,x2,y2):
    """
    Load and crop an image

    Parameters
    img_path (str): Path to the input image file.
    x1, y1, x2, y2 (int): Coordinates defining the top-left (x1,y1) and bottom-right (x2,y2) corners of the crop.

    Returns
    ndarray: Cropped image.
    """
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cropped_img = img_rgb[y1:y2, x1:x2]
    return cropped_img


def crop_and_preprocess_img(img_path, x1,y1,x2,y2, color_space="lab"):
    """
    Load, crop, and preprocess (using contrast
    enhancement and color space conversion) an image for segmentation.

    Parameters
    img_path (str): Path to the input image file.
    x1, y1, x2, y2 (int): Coordinates defining the top-left (x1,y1) and bottom-right (x2,y2) corners of the crop.

    Returns
    ndarray: Preprocessed image suitable for segmentation.
    """
    cropped_img = crop_image(img_path, x1,y1,x2,y2)
    preprocessed_img = preprocess_image(cropped_img, color_space)
    return preprocessed_img