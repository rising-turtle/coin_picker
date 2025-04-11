#!/usr/bin/env python

"""Multiple coins (circles) detection using OpenCV."""

__author__ = "Yina Tang"
__credits__ = ["Dr. He \"David\" Zhang"]

import inspect
import logging
import os
import requests
import sys
import time
from typing_extensions import Sequence
from helpers import resized, resizedImageFile
import cv2
from cv2.typing import MatLike
import numpy as np
from PIL import Image
from PIL.ImageFile import ImageFile
import skimage as ski
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour
# from sklearn import svm
import matplotlib.pyplot as plt

# Set up
directory: str = "/Users/inatang/Developer/coin_picker/detection"  # change this
os.chdir(directory)
logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(filename='detection.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')


def circles(image_path: str):
    """Finds circles in an image using HoughCircles

    HoughCircles: https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html

    Problems
    --------
    - finding min and max radius (normalize image size?)
    - removing duplicate circles !important
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert image is not None, f"Image not found at {image_path}"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 125, 175)  # parameters from trial and error
    # cv2.imshow("Edges", edges)
    # cv2.waitKey(0)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30)
    if circles is None:
        print("No circles found")
        return
    circles = np.around(circles).astype(np.uint16)
    print(circles)
    circled_image = image
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(circled_image, (i[0], i[1]), i[2], (0,255,0), 2)
        # draw the center of the circle
        cv2.circle(circled_image, (i[0], i[1]), 2, (0,0,255), 3)

        cv2.imshow('Circles', circled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_ellipses(image_path: str):
    """Detects and segments coins in an image using OpenCV.

    Returns a list of segmented coin images in RGB.

    Problem: Outputs lots of elliptical noises...
    """

    # def circle_contour() -> MatLike:
    #     """Helper function to create a circle contour for measuring the dissimilarity of a contour to a circle.

    #     ellipsePoly works better...I don't know why...
    #     """
    #     # 1. Create a black image (grayscale for simplicity)
    #     height, width = 400, 400
    #     img = np.zeros((height, width), dtype=np.uint8)

    #     # 2. Draw a white filled circle on the black image
    #     cv2.circle(img, center=(width // 2, height // 2), radius=50, color=(255,), thickness=-1)

    #     # 3. Find contours on the image
    #     contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     return contours[0]


    # Preprocess image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_out = image.copy()
    assert image is not None, f"Image not found at {image_path}"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Detect edges with Canny: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    edges = cv2.Canny(blurred, 125, 175)  # parameters from trial and error

    # Find contours: https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found")
        return image_out

    coins: list[MatLike] = []

    # arcLength works better than contourArea
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, closed=False), reverse=True)

    # Iterate through contours to find and segment the coins
    for contour in contours:

        # Must manually close the contour circle, fit circle, or fit ellipse to allow filling of contour for mask
        try:
            ellipse_fit = cv2.fitEllipse(contour)
        except Exception as e:
            print(f"Error fitting ellipse to contour: {e}")
            return image_out

        center, axes, angle = ellipse_fit

        # Convert center to integer and half the axes:
        center = (int(center[0]), int(center[1]))
        axes   = (int(axes[0] / 2), int(axes[1] / 2))
        angle  = int(angle)

        # Approximate the ellipse boundary by a polygon with 1-degree steps:
        # improve the ellipse approximation if we have time...
        ellipse_poly = cv2.ellipse2Poly(center, axes, angle, 0, 360, 1)
        # Turn the ellipse into a contour-like object for matchShapes():
        ellipse_matlike = np.array(ellipse_poly, dtype=np.float32).reshape((-1, 1, 2))

        # Determine if contour could be a coin by comparing to a circle:
        # TODO: determine best threshhold (train model?)
        threshhold = 2.0
        # TODO: determine which CONTOURS_MATCH_* method to use
        # I don't understand why matchshapes seems to work...so replace in future
        dissimilarity = cv2.matchShapes(contour, ellipse_matlike, cv2.CONTOURS_MATCH_I2, 0)
        if dissimilarity > threshhold:
            print(f"({len(coins)+1:0>2}) {dissimilarity:.2f} > {threshhold} dissimilarity , stop detecting")
            continue

        print(f"({len(coins)+1:0>2}) {dissimilarity:.2f} dissimilarity, segmenting coin...")

        # Draw ellipses
        cv2.ellipse(image_out, ellipse_fit, (0, 255, 0), thickness=2)
        # cv2.imshow('Coins', image_out)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    #     # Segment the coin by masking:
    #     mask = np.zeros_like(gray, dtype=np.uint8)
    #     cv2.ellipse(mask, ellipse_fit, (255, 255, 255), thickness=cv2.FILLED)
    #     masked_image = cv2.bitwise_and(image, image, mask=mask)  # RGB
    #     if masked_image is None:
    #         print("Error masking image")
    #         continue

    #     coins.append(masked_image)

    # return coins

    return image_out


# USE THIS!
def detect_circles(image_path: str, max_coins = 50, resize=True):
    """Detects and segments coins in an image using OpenCV.

    Returns a list of segmented coin images in RGB.
    """

    # Preprocess image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert image is not None, f"Image not found at {image_path}"

    if resize and (image.shape[0] > 720 or image.shape[1] > 1280):
        image = resized(image)

    image_out = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Detect edges with Canny: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    edges = cv2.Canny(blurred, 125, 175)  # parameters from trial and error
    cv2.imshow('Canny', edges)
    cv2.waitKey(0)

    # Find contours: https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found")
        return image_out

    # coins: list[MatLike] = []
    coin_edges = []

    # arcLength works better than contourArea
    # But arcLength is not a good measure when coins overlap since minEnclosingCircle gets too much noise then
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, closed=False), reverse=True)
    contours = contours[:max_coins]

    # IDEA: Draw contours and compare length with perimeter ot minEnclosingCircle (circularity)

    # Iterate through contours to find and segment the coins
    for contour in contours:

        # Must manually close the contour circle, fit circle, or fit ellipse to allow filling of contour for mask
        try:
            (x,y), radius = cv2.minEnclosingCircle(contour)
        except Exception as e:
            print(f"Error fitting ellipse to contour: {e}")
            return image_out

        # If center is within radius of any previous circles, skip
        # Works surprisingly well for most cases
        # TODO: improve noise removal on the edge of contours
        # TODO: improve overlapping minEnclosingCircles handling
        is_overlapping = False
        for coin_edge in coin_edges:
            (x_prev, y_prev), r_prev = coin_edge
            if abs(x-x_prev) < r_prev and abs(y-y_prev) < r_prev:
                is_overlapping = True
                break
        if is_overlapping: continue

        # If center is within some distance of the edge of any previous circles, skip

        coin_edges.append(((x,y), radius))
        cv2.circle(image_out, (int(x), int(y)), int(radius), (0,255,0), 2)

        # cv2.imshow('Coins', image_out)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    #     # Segment the coin by masking:
    #     mask = np.zeros_like(gray, dtype=np.uint8)
    #     cv2.ellipse(mask, ellipse_fit, (255, 255, 255), thickness=cv2.FILLED)
    #     masked_image = cv2.bitwise_and(image, image, mask=mask)  # RGB
    #     if masked_image is None:
    #         print("Error masking image")
    #         continue

    #     coins.append(masked_image)

    # return coins

    return image_out


def morph_gac(image_path: str, *, num_iter=250, timer=True) -> None:
    def store_evolution_in(lst):
        """Returns a callback function to store the evolution of the level sets in the given list."""
        def _store(x): lst.append(np.copy(x))
        return _store

    start_time: float = time.time()  # avoid pyright yelling at me for Unbound
    image_name: str = image_path.split('/')[-1].split('.')[0]

    # Set up
    # print("Setting up morphological GAC segmentation...")
    # image = np.asarray(resizedImageFile(Image.open(image_path)))  # resize image to 1280x720
    # gray = rgb2gray(image)
    # init_ls = np.zeros(gray.shape, dtype=np.int8)
    # init_ls[10:-10, 10:-10] = 1
    # evolution = []
    # callback = store_evolution_in(evolution)

    # print("Preprocessing image for MorphGAC...")
    # preprocessed: ImageFile = inverse_gaussian_gradient(ski.util.img_as_float(gray))

    # print("Running morphological GAC segmentation...")
    # ls = morphological_geodesic_active_contour(
    #     preprocessed_image,
    #     num_iter=num_iter,
    #     init_level_set=init_ls,
    #     smoothing=1,
    #     balloon=-1,
    #     threshold=0.69,
    #     iter_callback=callback,
    # )

    # ls = morphological_geodesic_active_contour(
    #     preprocessed_image,
    #     num_iter=num_iter,
    #     init_level_set=init_ls,
    #     smoothing=1,
    #     balloon=-1,
    #     threshold=0.69,
    #     iter_callback=callback,
    # )

    # sliding window
    # Set up
    print("Setting up morphological GAC segmentation...")
    image = np.asarray(resizedImageFile(Image.open(image_path)))  # resize image to 1280x720
    gray = rgb2gray(image)
    print(f"Image dimensions: {gray.shape}")

    print("Preprocessing image for MorphGAC...")
    preprocessed = inverse_gaussian_gradient(gray)
    # plt.imshow(preprocessed, cmap="gray")

    print("Creating windows for sliding...")
    xstep, ystep = 40, 40
    coin_width, coin_height = 80, 80
    # windows = [preprocessed[x:x+coin_width, y:y+coin_height] for x in range(0, preprocessed.shape[0]-coin_width, xstep) for y in range(0, preprocessed.shape[1]-coin_height, ystep)]
    # windows = np.asarray([preprocessed_imagefile.crop((x, y, x+coin_width, y+coin_height)) for x in range(preprocessed_imagefile.width-coin_width+1, xstep) for y in range(preprocessed_imagefile.height-coin_height+1, ystep)])

    fig, axes = plt.subplots(1, 1, figsize=(6, 8))
    ax = axes
    ax.imshow(image, cmap="gray")
    ax.set_axis_off()
    ax.set_title(f"{image_name}_morphgac{num_iter}", fontsize=12)
    plot = np.zeros(gray.shape, dtype=np.int8)

    print("Running morphological GAC segmentation...")
    # for i, window in enumerate(windows):
        # print(f"Window {i+1}/{len(windows)}: ")
    for x in range(0, preprocessed.shape[0]-coin_width, xstep):
        for y in range(0, preprocessed.shape[1]-coin_height, ystep):
            print(f"Window {(x, y)}")
            window = preprocessed[x:x+coin_width, y:y+coin_height]

            init_ls = np.zeros(window.shape, dtype=np.int8)
            init_ls[10:-10, 10:-10] = 1
            evolution = []
            callback = store_evolution_in(evolution)

            # Run morphological GAC segmentation on the window
            # Returns a binary mask of the segmented region (in this case 80 x 80)
            contour = morphological_geodesic_active_contour(
                window,
                num_iter=num_iter,
                init_level_set=init_ls,
                smoothing=1,
                balloon=-1,
                threshold=0.69,
                iter_callback=callback,
            )

            # Pad contour to size of image
            contour = np.pad(contour, ((x, gray.shape[0]-coin_width-x), (y, gray.shape[1]-coin_height-y)), mode='constant', constant_values=0)

            # Combine contours for plotting eventually
            plot = np.logical_or(contour, plot)

        # Get the coordinates of the top-left corner of window

            # xpad, ypad = i * xstep, i * ystep

            # # Pad the mask to size of image
            # print(f"beforeX: {xpad}, afterX: {gray.shape[0]-coin_width-xpad}, beforeY: {ypad}, afterY: {gray.shape[1]-coin_height-ypad}")
            # contour = np.pad(contour, ((xpad, gray.shape[0]-coin_width-xpad), (ypad, gray.shape[1]-coin_height-ypad)), mode='constant', constant_values=0)
            # plot = np.logical_or(contour, plot)

    ax.contour(plot, [0.5], colors='r')

    # Send notification to my phone through ntfy.sh
    print("Morphological GAC segmentation complete!")
    requests.post("https://ntfy.sh/i7t5", data="Morphological GAC segmentation complete!".encode(encoding="utf-8"))

    if timer:  # I should probably just remove timer and time every time...
        time_diff: float = time.time() - start_time
        time_str: str = f"{time_diff:.2f}s" if time_diff < 60 else f"{(time_diff/60):.2f}min"
        print(f"Active contours runtime: {time_str}")
        logger.info(f"{f'morph_gac{num_iter}'.ljust(14)} {image_name.ljust(20)} {time_str}")
        # could use inspect.currentframe().f_code.co_name to get current function name
        # or inspect.currentframe().f_back.f_code.co_name to get calling function name

    # print("Plotting...")
    # fig, axes = plt.subplots(1, 1, figsize=(6, 8))
    # ax = axes
    # ax.imshow(image, cmap="gray")
    # ax.set_axis_off()
    # ax.contour(ls, [0.5], colors='r')
    # ax.set_title(f"{image_name}_morphgac{num_iter}", fontsize=12)
    filename: str = f"{image_name}_morphgac{num_iter}.jpg"
    plt.savefig(os.path.join("out", "morph_gac", filename), bbox_inches='tight', pad_inches=0.1)
    plt.show()


image_path = 'images/coins-distinct-44.jpg'
# max_coins = 50
# resize = True
# image = detect_circles(image_path, max_coins)
# # coins = detect_ellipses(image_path)
# # if not coins:
# #     print("No coins found")
# #     sys.exit()

# cv2.imshow('Coins', image)

image_name = image_path.split('/')[-1].split('.')[0]
# filename = f'{image_name}_circle_max{max_coins}{"" if resize else "_noresize"}.jpg'
# cv2.imwrite(os.path.join("out", filename), image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for coin in coins:
#     cv2.imshow('Coin', coin)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

num_iter: int = 250
morph_gac(image_path, num_iter=num_iter)
# resizedImageFile(Image.open(image_path), outpath=f'{image_name}_resized.jpg')

"""
MorphGAC
Without resize:
- coins-touching-8_morphgac1000 with inverse_gaussian_gradient took 15 minutes ish?
- coins-overlapping-1_morphgac1000 with inverse_gaussian_gradient took 30 minutes...

With resize
- penny_head_morphgac1000 with inverse_gaussian_gradient took 25.55s
- coins-touching-8_morphgac1000 with inverse_gaussian_gradient took 46.76s
- coins-touching-8_morphgac2000 with inverse_gaussian_gradient took 1.49min
- coins-distinct-4_morphgac250 with inverse_gaussian_gradient took 14.28s

"""
