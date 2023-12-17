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
    feature_params = dict(maxCorners=200,
                        qualityLevel=0.01,
                        minDistance=30,
                        blockSize=9,
                        k=0.04)
    
    # find keypoints in the current frame
    previous_keypoints = cv2.goodFeaturesToTrack(previous_image, mask=None, useHarrisDetector=True, gradientSize=3, **feature_params)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(previous_image)
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
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


def estimate_pose(
    current_keypoints, associated_keypoints, camera_intrinsics, previous_pose
):
    """
    Estimates the current camera pose based on keypoint associations.

    :param current_keypoints: Keypoints in the current frame.
    :param associated_keypoints: Corresponding keypoints in the previous frame.
    :param camera_intrinsics: Camera intrinsic parameters.
    :param previous_pose: The previous pose of the camera.
    :return: The estimated current pose of the camera.
    """
    pass


def triangulate_landmarks(keypoints, pose, camera_intrinsics):
    """
    Triangulates new landmarks based on keypoints and the current camera pose.

    :param keypoints: Keypoints for which landmarks need to be triangulated.
    :param pose: The current pose of the camera.
    :param camera_intrinsics: Camera intrinsic parameters.
    :return: A list of new triangulated landmarks.
    """
    pass


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
