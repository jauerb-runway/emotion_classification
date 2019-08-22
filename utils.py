"""Useful utility functions for emotion classification."""

import numpy as np
import cv2


def cv2_to_runway(bounding_box, image_width, image_height):
    """Convert OpenCV bounding box to Runway format.

    Args:
        bounding_box: bounding box in OpenCV format: (x, y, width, height)
            in image coordinates
        image_width: width of image (number of pixels)
        image_height: height of image (number of pixels)

    Returns:
        Runway format bounding box (xmin, ymin, xmax, ymax) in [0,1]
            where (xmin, ymin) is the top-left corner of the rectangle and
            (xmax, ymax) is the bottom-right corner.

    """
    (x, y, width, height) = bounding_box
    return (x/image_width, y/image_height,
            (x + width)/image_width, (y + height)/image_height)


def runway_to_cv2(bounding_box, image_width, image_height):
    """Convert Runway bounding box to OpenCV format.

    Args:
        bounding_box: bounding box in Runway format:
            (xmin, ymin, xmax, ymax) in [0,1]
            where (xmin, ymin) is the top-left corner of the rectangle and
            (xmax, ymax) is the bottom-right corner.
        image_width: width of image (number of pixels)
        image_height: height of image (number of pixels)

    Returns:
        bounding box in OpenCV format: (x, y, width, height)
            in image coordinates

    """
    (xmin, ymin, xmax, ymax) = bounding_box
    return (round(xmin * image_width), round(ymin * image_height),
            round((xmax - xmin) * image_width),
            round((ymax - ymin) * image_height))


def grayscale_from_pil(pil_image):
    """Convert a PIL image to a 2D numpy array (grayscale).

    Args:
        pil_image: PIL image in RGB

    Returns:
        2D numpy array

    """
    np_image = np.array(pil_image)
    return cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)


def extract_roi(image, bounding_box):
    """Extract region of interest from image defined by bounding_box.

    Args:
        image: grayscale image as 2D numpy array of shape (height, width)
        bounding_box: (x, y, width, height) in image coordinates

    Returns:
        region of interest as 2D numpy array

    """
    (x, y, width, height) = bounding_box
    return image[y:y + height, x:x + width]
