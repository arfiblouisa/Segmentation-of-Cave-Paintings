import argparse
from pathlib import Path
import sys
from networkx import radius
from skimage.morphology import remove_small_objects
from utils import *
import seg_cave_paintings.preprocessing_utils as prep_utils
import seg_cave_paintings.meanshift as ms_utils
import seg_cave_paintings.morphological_active_contours as mgac_utils
import seg_cave_paintings.otsu as otsu_utils
from ui.roi_selector import select_roi
from ui.region_merger import *


def run_meanshift(args):
    print("MeanShift selected")
    img = load_image(args.image)
    x1, y1, x2, y2=select_roi(img)

    preprocessed_img=prep_utils.crop_and_preprocess_img(args.image, x1, y1, x2, y2)

    print("Applying Mean Shift algorithm...")

    labels_img, segmented, result = ms_utils.apply_meanshift(
        preprocessed_img,
        spatial_radius=args.spatial_radius,
        color_radius=args.color_radius
    )

    selected_regions, regions = interactive_region_selection(
        segmented,
        labels_img
    )

    mask_cropped = ms_utils.create_segmentation_results(
        selected_regions,
        labels_img,
        preprocessed_img,
        save_path=args.save
    )



def run_snake(args):
    print("Snake selected")
    img = load_image(args.image)
    x1, y1, x2, y2=select_roi(img)

    preprocessed_img=prep_utils.crop_and_preprocess_img(args.image, x1, y1, x2, y2)

    print("Applying active contours...")
    snake_mask, evolution=mgac_utils.apply_active_contours(preprocessed_img, num_iter=args.num_iter, threshold=args.threshold) # Apply active contours to segment the image

    h,w=snake_mask.shape
    snake_mask_filtered = remove_small_objects(snake_mask.astype(bool), min_size=h*w/100)

    mgac_utils.create_segmentation_results(snake_mask_filtered, preprocessed_img, save_path=args.save)



def run_otsu(args):
    print("Otsu thresholding selected")
    img = load_image(args.image)
    x1, y1, x2, y2=select_roi(img)

    preprocessed_img=prep_utils.crop_image(args.image, x1,y1,x2,y2)

    mask_horse = otsu_utils.otsu_mask_from_color_space(preprocessed_img)
    mask_horse = mask_horse > 0
    otsu_utils.create_segmentation_results(mask_horse, preprocessed_img, save_path=args.save)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive cave painting segmentation tool"
    )

    subparsers = parser.add_subparsers(
        dest="algorithm",
        required=True,
        help="Segmentation algorithm"
    )

    meanshift_parser = subparsers.add_parser(
        "meanshift",
        help="Use Mean Shift segmentation"
    )

    meanshift_parser.add_argument("--image", required=True)
    meanshift_parser.add_argument("--spatial_radius", type=int, default=80)
    meanshift_parser.add_argument("--color_radius", type=int, default=24)
    meanshift_parser.add_argument("--save", type=str, default=None)


    snake_parser = subparsers.add_parser(
        "active_contours",
        help="Use Active Contours (Snake)"
    )

    snake_parser.add_argument("--image", required=True)
    snake_parser.add_argument("--num_iter", type=int, default=500)
    snake_parser.add_argument("--threshold", type=float, default=0.75)
    snake_parser.add_argument("--save", type=str, default=None)


    otsu_parser = subparsers.add_parser(
        "otsu",
        help="Use Otsu thresholding"
    )

    otsu_parser.add_argument("--image", required=True)
    otsu_parser.add_argument("--save", type=str, default=None)


    meanshift_parser.set_defaults(func=run_meanshift)
    snake_parser.set_defaults(func=run_snake)
    otsu_parser.set_defaults(func=run_otsu)


    args = parser.parse_args()
    

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()