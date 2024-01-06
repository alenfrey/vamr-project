import numpy as np
import cv2
from src.utils import timer
from collections import namedtuple

Features = namedtuple("Features", ["keypoints", "descriptors"])


def detect_features(detector, img):
    """Detects features in an image"""
    keypoints, descriptors = detector.detectAndCompute(img, None)
    return Features(keypoints, descriptors)

@timer
def match_features(matcher, features_a, features_b, lowe_ratio=0.8, max_distance=30):
    """Detects and matches features between two images with cross-check and distance constraint"""
    keypoints_a, descriptors_a = features_a
    keypoints_b, descriptors_b = features_b

    def cross_check_matches(matches_ab, matches_ba):
        """Performs a cross-check between matches."""
        mutual_matches = []
        for match_ab in matches_ab:
            if matches_ba[match_ab.trainIdx].trainIdx == match_ab.queryIdx:
                mutual_matches.append(match_ab)
        return mutual_matches

    # Forward matching
    matches_ab = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    matches_ab = [m for m, n in matches_ab if m.distance < lowe_ratio * n.distance]

    # Reverse matching
    matches_ba = matcher.knnMatch(descriptors_b, descriptors_a, k=1)
    matches_ba = [m[0] for m in matches_ba]

    # Cross-check
    good_matches = cross_check_matches(matches_ab, matches_ba)

    # Apply geometric constraints
    filtered_matches = []
    for m in good_matches:
        pt_a = keypoints_a[m.queryIdx].pt
        pt_b = keypoints_b[m.trainIdx].pt
        distance = np.linalg.norm(np.array(pt_a) - np.array(pt_b))
        if distance <= max_distance:
            filtered_matches.append(m)

    print(
        f"Number of good matches after cross-check and distance filtering: {len(filtered_matches)}"
    )

    # Extracting the point coordinates
    pts_a = np.float32([keypoints_a[m.queryIdx].pt for m in filtered_matches]).reshape(
        -1, 1, 2
    )
    pts_b = np.float32([keypoints_b[m.trainIdx].pt for m in filtered_matches]).reshape(
        -1, 1, 2
    )

    return pts_a, pts_b, filtered_matches


def generate_match_image(img_a, img_b, features_a, features_b, good_matches):
    """Shows an image showing the matches between two images"""
    match_img = cv2.drawMatches(
        img_a,
        features_a.keypoints,
        img_b,
        features_b.keypoints,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return match_img

@timer
def estimate_pose(keypoints_a, keypoints_b, K, prob=0.995, threshold=1.0):
    """Estimates the pose between two images using RANSAC"""
    E, inlier_mask = cv2.findEssentialMat(
        keypoints_a,
        keypoints_b,
        K,
        method=cv2.LMEDS,
        prob=prob,
        threshold=threshold,
        # maxIters=3000,
    )

    # filter points based on the inlier mask
    pts_a_inliers = keypoints_a[inlier_mask.ravel() == 1]
    pts_b_inliers = keypoints_b[inlier_mask.ravel() == 1]

    print(f"Initial number of inlier matches: {len(pts_a_inliers)}")
    print(f"Initial number of outlier matches: {len(keypoints_a) - len(pts_a_inliers)}")

    # recover camera pose and refine inliers
    _, R, t, pose_mask = cv2.recoverPose(E, pts_a_inliers, pts_b_inliers, K)

    pose_mask = (pose_mask > 0).astype(int)
    # Further filter points based on the pose mask
    pts_a_inliers_refined = pts_a_inliers[pose_mask.ravel() == 1]
    pts_b_inliers_refined = pts_b_inliers[pose_mask.ravel() == 1]

    print(f"refined number of inlier matches: {len(pts_a_inliers_refined)}")
    print(
        f"refined number of outlier matches: {len(keypoints_a) - len(pts_a_inliers_refined)}"
    )

    return R, t, pts_a_inliers_refined, pts_b_inliers_refined

@timer
def triangulate_points(pts_a, pts_b, K, relative_pose):
    """Triangulates points from two images"""
    R = relative_pose[:3, :3]  # rotation matrix from the relative pose
    t = relative_pose[:3, 3:4]  # translation vector from the relative pose
    P0 = np.hstack(
        (np.eye(3), np.zeros((3, 1)))
    )  # projection matrix for the first camera
    P1 = np.hstack((R, t))  # Projection matrix for the second camera
    P0 = K @ P0  # apply the intrinsic matrix to the first camera
    P1 = K @ P1  # apply the intrinsic matrix to the second camera
    pts4D = cv2.triangulatePoints(P0, P1, pts_a, pts_b)
    pts3D = pts4D[:3] / pts4D[3]  # convert from homogeneous to 3D coordinates

    # in_front_of_camera = np.logical_and(
    #     (P0 @ pts4D)[-1] < 0,  # Z-coordinate in camera 0's coordinate system
    #     (P1 @ pts4D)[-1] < 0,  # Z-coordinate in camera 1's coordinate system
    # )
    # pts3D = pts3D[:, in_front_of_camera]
    return pts3D
