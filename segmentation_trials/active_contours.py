import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from scipy.interpolate import splprep, splev
import cv2
from skimage.filters import sobel

# Charger l'image
img_bgr = cv2.imread("Segmentation-of-Cave-Paintings\dataset\cosquer\panneau_des_cheveaux.jpg")  # image couleur ou niveaux de gris
img_rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
img_l=img_lab[:,:,0]  # canal L (luminosité)


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

def preprocess_charcoal_image(img):
    
    out = convert_color_space(img, "lab")
    out = clahe_image(out)
        
    return out

preprocessed_img=preprocess_charcoal_image(img_rgb)

img_smooth = gaussian(preprocessed_img, sigma=3)
fx = sobel(img_smooth, axis=1)
fy = sobel(img_smooth, axis=0)
edges = np.sqrt(fx**2 + fy**2)
g = 1 / (1 + edges**2)


# Affichage pour sélection des points
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img_l, cmap="gray")
ax.set_title("Cliquez pour définir le contour initial\nClic droit ou Entrée pour terminer")
ax.axis("off")

# Sélection interactive
points = plt.ginput(n=-1, timeout=0)  # n=-1 => nombre libre de points
plt.close(fig)

points = np.array(points)  # shape (N, 2) avec (x, y)

# Fermer le contour
points = np.vstack([points, points[0]])

# Interpolation spline fermée
tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=True)
u_new = np.linspace(0, 1, 400)  # nombre de points du contour initial
x_new, y_new = splev(u_new, tck)

# Format attendu par active_contour : (row, col)
init = np.array([y_new, x_new]).T

# snake = active_contour(
#     gaussian(img_l, sigma=3, preserve_range=False),
#     init,
#     alpha=0.05,
#     beta=10,
#     gamma=0.001,
#     w_line=-1,
#     w_edge=1,
# )
# snake = active_contour(
#     g,
#     init,
#     alpha=0.001,
#     beta=50,
#     gamma=0.001,
#     max_num_iter=400,
# )
snake = active_contour(
    gaussian(preprocessed_img, sigma=2),
    init,
    alpha=1,
    beta=1,
    gamma=0.01,
    w_line=-1,
    w_edge=1,
    max_num_iter=1000,
)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img_rgb)
ax.plot(init[:, 1], init[:, 0], '--r', lw=2, label="Initial (utilisateur)")
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=2, label="Snake final")
ax.legend()
ax.axis([0, img_rgb.shape[1], img_rgb.shape[0], 0])
ax.set_xticks([]), ax.set_yticks([])
plt.show()
