"""
This file contains the initialization steps for the VO pipeline.
"""

import cv2
import numpy as np

from src.utils import timer


def ensure_grayscale(image):
    """
    Convert an image to grayscale if it is not already.
    """
    # return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) -> Alternative short
    # check if the image has more than one channel (i.e., is not grayscale)
    if len(image.shape) > 2 and image.shape[2] > 1:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # the image is already grayscale
        return image


# EXAMPLE OF AN INITIALIZATION FUNCTION
@timer
def initialize_vo(initialization_images, camera_intrinsics):
    # use ORB feature detector and matcher
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # detect and compute features (first image)
    keypoints_1, descriptors_1 = orb.detectAndCompute(initialization_images[0], None)

    # detect and compute features (second image)
    keypoints_2, descriptors_2 = orb.detectAndCompute(initialization_images[1], None)

    # match features between the two images
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    # wxtract matched keypoints
    points_1 = np.float32([keypoints_1[m.queryIdx].pt for m in matches])
    points_2 = np.float32([keypoints_2[m.trainIdx].pt for m in matches])

    # find the Essential matrix
    E, mask = cv2.findEssentialMat(
        points_1, points_2, camera_intrinsics, cv2.RANSAC, 0.999, 1.0, None
    )

    # decompose essentiall matrix into R, t (pose)
    _, R, t, _ = cv2.recoverPose(E, points_1, points_2, camera_intrinsics)

    # TODO: initialize landmarks (3D points) (triangulatoin)

    return R, t  # Return initial rotation and translation matrices
