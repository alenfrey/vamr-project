"""
This file contains the functions used in the VO pipeline during continuous operation.
"""

from src.utils import timer
import cv2
import numpy as np


@timer
def associate_keypoints(current_image, previous_image):
    """
    Associates keypoints in the current frame with those in the previous frame.

    :param current_image: The current frame as a numpy array.
    :param previous_image: The previous frame as a numpy array.
    :return: A tuple of (annotated_image, current_keypoints, associated_keypoints), where
             current_keypoints are keypoints in the current frame, and associated_keypoints
             are the corresponding keypoints from the previous frame. Lastly, annotated_image
             is the current frame with the associated keypoints drawn on it.
    """
    # initialize harris corner detector parameters
    feature_params = dict(
        maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=9, k=0.04
    )

    # find keypoints in the current frame
    previous_keypoints = cv2.goodFeaturesToTrack(
        previous_image,
        mask=None,
        useHarrisDetector=True,
        gradientSize=3,
        **feature_params,
    )

    # Create a mask image for drawing purposes
    mask = np.zeros_like(previous_image)

    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # calculate optical flow
    curr_keypoints, status, error = cv2.calcOpticalFlowPyrLK(
        previous_image, current_image, previous_keypoints, None, **lk_params
    )

    # select only the keypoints that were successfully tracked
    if curr_keypoints is not None:
        curr_keypoints = curr_keypoints[status == 1]
        prev_keypoints = previous_keypoints[status == 1]

    # draw the tracks
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    for i, (new, old) in enumerate(zip(curr_keypoints, prev_keypoints)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(current_image, (int(a), int(b)), 5, color[i].tolist(), -1)
    annotated_image = cv2.add(frame, mask)
    return annotated_image, curr_keypoints, prev_keypoints


def find_2D_to_3D_correspondences(
    points_3D,
    keypoints_2D,
    rotation_vector,
    translation_vector,
    camera_intrinsics,
    dist_coeffs=np.zeros((4, 1)),
):
    # Project 3D points to 2D plane
    projected_points, _ = cv2.projectPoints(
        points_3D, rotation_vector, translation_vector, camera_intrinsics, dist_coeffs
    )

    # Matching keypoints
    matched_keypoints_2D = []
    matched_points_3D = []

    for i, projected_point in enumerate(projected_points):
        x, y = projected_point.ravel()
        # Find the nearest 2D keypoint
        distances = np.linalg.norm(keypoints_2D - np.array([x, y]), axis=1)
        min_distance_index = np.argmin(distances)
        if distances[min_distance_index] < 8:  # Define a suitable threshold
            matched_keypoints_2D.append(keypoints_2D[min_distance_index])
            matched_points_3D.append(points_3D[i])

    return matched_keypoints_2D, matched_points_3D


def estimate_pose(
    keypoints_2D, points_3D, camera_intrinsics, dist_coeffs=np.zeros((4, 1))
):
    assert len(keypoints_2D) == len(
        points_3D
    ), "The number of 2D keypoints must match the number of 3D points"

    keypoints_2D = np.array(keypoints_2D, dtype=np.float32)
    points_3D = np.array(points_3D, dtype=np.float32)

    # Use solvePnPRansac to estimate pose
    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        points_3D, keypoints_2D, camera_intrinsics, dist_coeffs
    )

    if not success:
        raise Exception("Pose estimation failed.")

    R, _ = cv2.Rodrigues(rotation_vector)
    
    return R, translation_vector, inliers


def triangulate_landmarks(keypoints_1, keypoints_2, R, t, camera_intrinsics):
    # Ensure R is 3x3 and t is 3x1
    assert R.shape == (3, 3), "R must be a 3x3 matrix"
    assert t.shape == (3, 1), "t must be a 3x1 vector"

    # Constructing the 3x4 projection matrices
    P0 = np.dot(camera_intrinsics, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P1 = np.dot(camera_intrinsics, np.hstack((R, t)))

    # Ensure keypoints are in the correct shape (2xN)
    if keypoints_1.shape[0] != 2:
        keypoints_1 = keypoints_1.reshape(2, -1)
    if keypoints_2.shape[0] != 2:
        keypoints_2 = keypoints_2.reshape(2, -1)

    # Triangulate points
    points_4D_hom = cv2.triangulatePoints(P0, P1, keypoints_1, keypoints_2)

    # Convert from homogeneous to 3D coordinates
    points_3D = points_4D_hom[:3, :] / np.tile(points_4D_hom[3, :], (3, 1))
    points_3D = points_3D.T

    return points_3D


def update_map_and_state(map, state, new_landmarks, current_pose):
    """
    Updates the map and the state of the VO system.

    :param map: The current map of the environment.
    :param state: The current state of the VO system.
    :param new_landmarks: New landmarks to be added to the map.
    :param current_pose: The current pose of the camera.
    :return: Updated map and state.
    """
    pass


def visualize_vo(image, keypoints, trajectory, landmarks):
    """
    Visualizes the current state of the VO system.

    :param image: The current frame.
    :param keypoints: Current detected keypoints.
    :param trajectory: The trajectory of the camera.
    :param landmarks: The landmarks in the map.
    """
    pass
