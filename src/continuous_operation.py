"""
This file contains the functions used in the VO pipeline during continuous operation.
"""

from src.utils import timer


@timer
def associate_keypoints(current_image, previous_image, previous_keypoints):
    """
    Associates keypoints in the current frame with those in the previous frame.

    :param current_image: The current frame as a numpy array.
    :param previous_image: The previous frame as a numpy array.
    :param previous_keypoints: Keypoints detected in the previous frame.
    :return: A tuple of (current_keypoints, associated_keypoints), where
             current_keypoints are keypoints in the current frame, and
             associated_keypoints are the corresponding keypoints from the previous frame.
    """
    pass


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
