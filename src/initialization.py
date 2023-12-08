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

@timer
def initialize_keypoints_harris(first_image, block_size, ksize, k):
    # Harris corner detectore for initializing keypoints
    img = first_image
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = img

    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)

    # Convert the grayscale image to a BGR color image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Iterate through each point and draw a cross
    for i in range(len(res)):
        y1, x1 = int(res[i, 1]), int(res[i, 0])
        y2, x2 = int(res[i, 3]), int(res[i, 2])

        # Draw yellow crosses
        draw_cross(img_color, (x1, y1))
        draw_cross(img_color, (x2, y2))

    # Display the result
    cv2.imshow('Image with Yellow Crosses', img_color)
    return corners

@timer
def initialize_keypoints_sift(first_image):
    # calcualte SIFT keypoints and descriptors
    sift = cv2.SIFT_create()

    # detect keypoints
    keypoints_1, descriptors_1 = sift.detectAndCompute(first_image, None)

    # Draw keypoints for visualization
    image1_with_keypoints = cv2.drawKeypoints(first_image, keypoints_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the image with keypoints
    cv2.imshow('Keypoints in the First Image', image1_with_keypoints)

    return image1_with_keypoints, keypoints_1, descriptors_1

# Function to draw a cross
def draw_cross(img, center, size=5, color=(0, 255, 255)):
    # Draw two lines intersecting at the center
    cv2.line(img, (center[0] - size, center[1]), (center[0] + size, center[1]), color, 1)
    cv2.line(img, (center[0], center[1] - size), (center[0], center[1] + size), color, 1)
