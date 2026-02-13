import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import gaussian, sobel
from skimage.segmentation import morphological_geodesic_active_contour
from skimage.draw import polygon
from scipy.interpolate import splprep, splev
from skimage.measure import find_contours
# Charger image
img_bgr = cv2.imread(
    "Segmentation-of-Cave-Paintings/dataset/cosquer/panneau_des_cheveaux.jpg"
)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

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
    # si lab -> canal L
    if out.ndim == 3:
        out = out[:,:,0]
    return out

preprocessed_img = preprocess_charcoal_image(img_bgr)
preprocessed_img = preprocessed_img.astype(np.float32) / 255.0

# ----------------------------
# Edge-stopping function robuste
# ----------------------------
img_smooth = gaussian(preprocessed_img, sigma=1)
edges = sobel(img_smooth)
print(edges.shape)
edges = edges / edges.max()
gimage = np.exp(-1 * edges)#initialement : 5

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(preprocessed_img, cmap='gray')
ax.set_title("Cliquez pour définir le contour initial\nEntrée ou clic droit pour terminer")
ax.axis('off')
points = plt.ginput(n=-1, timeout=0)
plt.close(fig)

points = np.array(points)
points = np.vstack([points, points[0]])

# Spline fermée pour lisser
tck, _ = splprep([points[:,0], points[:,1]], s=5, per=True)
u_new = np.linspace(0,1,400)
x_s, y_s = splev(u_new, tck)

# ----------------------------
# Masque initial pour MGAC
# ----------------------------
init_ls = np.zeros(preprocessed_img.shape, dtype=np.int8)
rr, cc = polygon(y_s.astype(int), x_s.astype(int), shape=preprocessed_img.shape)
init_ls[rr, cc] = 1

# ----------------------------
# Morphological Geodesic Active Contour
# ----------------------------
snake_mask = morphological_geodesic_active_contour(
    gimage,
    num_iter=200,
    init_level_set=init_ls,
    smoothing=3,
    balloon=-1   # pousse vers l'intérieur
)

# ----------------------------
# Extraction et affichage
# ----------------------------
contours = find_contours(snake_mask, level=0.5)
fig, ax = plt.subplots(figsize=(7,7))
ax.imshow(img_rgb)
for contour in contours:
    ax.plot(contour[:,1], contour[:,0], '-b', lw=2)
ax.set_title("MGAC après preprocessing")
ax.axis('off')
plt.show()