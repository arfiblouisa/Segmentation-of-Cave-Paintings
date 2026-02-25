import cv2

#Allows you to get the coordinates of bounding boxes drawn on images.
# Usage: write the path of the images you want to annotate in the "image_paths" list and run the script. Click and drag on the image to draw rectangles. The coordinates of each rectangle will be printed in the console. Press ENTER to move to the next image or ESC to skip.

drawing = False
x_start, y_start = -1, -1
x_end, y_end = -1, -1
img_copy = None


def draw_rectangle(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, drawing, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_temp = img_copy.copy()
        cv2.rectangle(img_temp, (x_start, y_start), (x, y), (0, 255, 0), 2)
        cv2.imshow("Image", img_temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_end, y_end = x, y
        cv2.rectangle(img_copy, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow("Image", img_copy)
        print(f"Rectangle: (x1={x_start}, y1={y_start}) -> (x2={x_end}, y2={y_end})")


def annotate_images(image_paths):
    global img_copy

    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)

        if img is None:
            print(f"Erreur : impossible de lire {path}")
            continue

        img_copy = img.copy()

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", draw_rectangle)

        print(f"\nImage {idx + 1}/{len(image_paths)} : {path}")
        print("Dessine un rectangle avec la souris.")

        cv2.imshow("Image", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_paths = [
        "dataset/cosquer/panneau_des_cheveaux.jpg",
        "dataset/mains/Cueva_de_las_manos_-_detalle.jpg",
        "dataset/lascaux/Lascaux_021.jpg"
    ]

    annotate_images(image_paths)
