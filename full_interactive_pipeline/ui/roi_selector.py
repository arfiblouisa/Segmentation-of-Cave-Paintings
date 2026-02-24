import cv2
import sys


def select_roi(image, max_dim=1000):
    """
    Displays a resized version of the image if too large.
    Returns ROI coordinates in ORIGINAL image scale.
    """

    print("Draw a bounding box around the painting and press ENTER.")
    print("Press 'c' to cancel.")

    h, w = image.shape[:2]

    # Compute scale factor
    scale = min(max_dim / max(h, w), 1.0)

    if scale < 1.0:
        resized = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )
    else:
        resized = image.copy()

    roi = cv2.selectROI(
        "Select Painting Region",
        resized,
        showCrosshair=False,
        fromCenter=False
    )

    cv2.destroyWindow("Select Painting Region")

    x, y, rw, rh = roi

    if rw == 0 or rh == 0:
        print("[INFO] No region selected. Exiting.")
        sys.exit(0)

    # Map back to original coordinates
    x1 = int(x / scale)
    y1 = int(y / scale)
    x2 = int((x + rw) / scale)
    y2 = int((y + rh) / scale)

    return x1, y1, x2, y2