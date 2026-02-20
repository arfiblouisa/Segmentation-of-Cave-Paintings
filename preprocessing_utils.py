import cv2

def convert_color_space(img, color_space):
    if color_space == "rgb":
        return img
    if color_space == "gray":
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if color_space == "hsv":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if color_space == "lab":
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    raise ValueError(color_space)


def equalize_image(img):
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ch = cv2.split(img)
    ch = [cv2.equalizeHist(c) for c in ch]
    return cv2.merge(ch)


def clahe_image(img, clip=2.0, grid=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    if img.ndim == 2:
        return clahe.apply(img)
    ch = cv2.split(img)
    ch = [clahe.apply(c) for c in ch]
    return cv2.merge(ch)


def denoise(img, method=None, g_sigma=1.0, m_ksize=2):
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


def crop_and_preprocess_img(img_path, x1,y1,x2,y2, color_space="lab"):
    cropped_img = crop_image(img_path, x1,y1,x2,y2)
    preprocessed_img = preprocess_image(cropped_img, color_space)
    return preprocessed_img


def crop_image(img_path, x1,y1,x2,y2):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cropped_img = img_rgb[y1:y2, x1:x2]
    return cropped_img

def preprocess_image(img, color_space="lab"):
    """ img must be RGB"""
    out = convert_color_space(img, color_space)
    out = clahe_image(out)
    out = denoise(out, "median", m_ksize=2)
    return out