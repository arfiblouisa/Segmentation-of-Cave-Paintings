import cv2
import numpy as np

def build_region_masks(labels_img):
    regions = {}
    for lab in np.unique(labels_img):
        regions[lab] = (labels_img == lab)
    return regions

def interactive_region_selection(image, labels):

    regions = build_region_masks(labels)
    selected_regions = set()
    current_hover = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_hover, selected_regions

        if y >= labels.shape[0] or x >= labels.shape[1]:
            return

        region_id = labels[y, x]

        if event == cv2.EVENT_MOUSEMOVE:
            current_hover = region_id

        elif event == cv2.EVENT_LBUTTONDOWN:
            if region_id in selected_regions:
                selected_regions.remove(region_id)
            else:
                selected_regions.add(region_id)

    cv2.namedWindow("Region Selection")
    cv2.setMouseCallback("Region Selection", mouse_callback)

    while True:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        overlay = bgr_image.copy()

        # Highlight hover region (yellow)
        if current_hover is not None:
            overlay[regions[current_hover]] = [0, 255, 255]

        # Highlight selected regions (red)
        for r in selected_regions:
            overlay[regions[r]] = [0, 0, 255]

        display = cv2.addWeighted(bgr_image, 0.6, overlay, 0.4, 0)

        cv2.imshow("Region Selection", display)

        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # ENTER
            break
        elif key == 27:  # ESC
            selected_regions.clear()
            break

    cv2.destroyWindow("Region Selection")

    return selected_regions, regions


def merge_selected_regions(selected_regions, regions, shape):
    final_mask = np.zeros(shape[:2], dtype=np.uint8)

    for r in selected_regions:
        final_mask[regions[r]] = 255

    return final_mask