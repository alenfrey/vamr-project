import cv2
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

from constants import *
from src.camera_calibration import *
from src.initialization import (
    initialize_vo,
)
from src.utils import *
from src.data_loaders import ParkingDataLoader
from src.continuous_operation import (
    associate_keypoints,
    estimate_pose,
    triangulate_landmarks,
    find_2D_to_3D_correspondences,
    update_points_3d,
)
from src.performance_metrics import FPSCounter, calculate_pose_error

# Define a threshold for minimum old points
MIN_OLD_POINTS_THRESHOLD = 50

fps_counter = FPSCounter()
dataset_loader = ParkingDataLoader(
    PARKING_DATA_DIR_PATH,
    init_frame_indices=[0, 1, 2],  # (frame indices for initialization)
    image_type=cv2.IMREAD_GRAYSCALE,  # can also use cv2.IMREAD_COLOR for color images
)

camera_intrinsics = dataset_loader.load_camera_intrinsics()
initialization_images = dataset_loader.get_initialization_frames()
R, t, points_3d = initialize_vo(initialization_images, camera_intrinsics)

projected_points, _ = cv2.projectPoints(
    points_3d, R, t, camera_intrinsics, distCoeffs=None
)

prev_image = initialization_images[-1]
prev_matched_keypoints_2D = None

for iteration, (curr_image, actual_pose, image_index) in enumerate(dataset_loader):
    print(f"pose: {actual_pose}")
    print(f"Processing frame {image_index}...")

    curr_image, curr_keypoints, prev_keypoints = associate_keypoints(
        curr_image, prev_image
    )

    # Check if keypoints are found
    if curr_keypoints is None or len(curr_keypoints) == 0:
        print("No keypoints found in the current frame.")
        continue  # Skip to the next frame

    # 2D to 3D Correspondences
    try:
        matched_keypoints_2D, matched_points_3D = find_2D_to_3D_correspondences(
            points_3d, curr_keypoints, R, t, camera_intrinsics
        )
    except Exception as e:
        print(f"Error in finding 2D-3D correspondences: {e}")
        continue  # Skip to the next frame

    # Check if valid correspondences are found
    if not matched_keypoints_2D or not matched_points_3D:
        print("No valid 2D-3D correspondences found.")
        continue  # Skip to the next frame

    try:
        R, t, inliers = estimate_pose(
            matched_keypoints_2D, matched_points_3D, camera_intrinsics
        )
    except Exception as e:
        print(f"Error in pose estimation: {e}")
        continue

    prev_image = curr_image.copy()
    # new_points_3D = triangulate_landmarks(
    #     prev_keypoints, curr_keypoints, R, t, camera_intrinsics
    # )

    # points_3d = (
    #     np.vstack([points_3d, new_points_3D]) if points_3d.size else new_points_3D
    # )
    
    # Count the number of matched old points in the current frame
    if prev_matched_keypoints_2D is not None:
        matched_old_points_count = len(prev_matched_keypoints_2D)
    else:
        matched_old_points_count = 0

    # Check if the number of matched old points is below the threshold
    if matched_old_points_count < MIN_OLD_POINTS_THRESHOLD:
        # Triangulate new 3D landmarks
        print(f"Number of matched old points: {matched_old_points_count}.")
        print("Triangulating new 3D landmarks...")
        new_points_3D = triangulate_landmarks(
            prev_keypoints, curr_keypoints, R, t, camera_intrinsics
        )

        # Add the new 3D points to the existing points
        points_3d = (
            np.vstack([points_3d, new_points_3D]) if points_3d.size else new_points_3D
        )
        
        # Update points_3d with new points and maintain size limit
        points_3d = update_points_3d(points_3d, new_points_3D)    
    
    prev_matched_keypoints_2D = matched_keypoints_2D.copy()
    

    estimated_pose = construct_homogeneous_matrix(R, t)
    print(f"Homogeneous estimated pose: {estimated_pose}")
    position_error, angle_error = calculate_pose_error(estimated_pose, actual_pose)

    print(f"Position error: {position_error}")
    print(f"Angle error: {angle_error}")

    fps = fps_counter.update()
    fps_counter.put_fps_on_image(curr_image, fps)

    cv2.imshow("Image Stream", curr_image)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break


# When everything is done, release the OpenCV window
cv2.destroyAllWindows()