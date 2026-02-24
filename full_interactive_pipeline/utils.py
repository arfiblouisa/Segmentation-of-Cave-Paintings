import cv2
import sys

def load_image(path):
    img = cv2.imread(path)

    if img is None:
        print(f"[ERROR] Could not load image: {path}")
        sys.exit(1)

    return img


def crop_image(image, roi):
    x, y, w, h = roi
    return image[y:y+h, x:x+w]


