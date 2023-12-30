import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils import *
from src.data_loaders import ParkingDataLoader, KittiDataLoader
from src.visualization import *
from src.performance_metrics import calculate_reprojection_error

# SIFT feature detector and FLANN matcher
sift_detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=50, sigma=1.6)
flann_matcher = cv2.FlannBasedMatcher()
visualizer = VOVisualizer()

# Data loader
dataset_loader = ParkingDataLoader(init_frame_indices=[0, 2])

# Helper function to extract keypoints
extract_keypoints = lambda kpts1, kpts2, matches: (
    np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
    np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
)

# Load camera intrinsics and initialization images
camera_intrinsics = dataset_loader.load_camera_intrinsics()
initialization_images = dataset_loader.get_initialization_frames()
prev_image = initialization_images[-1]
prev_keypoints, prev_descriptors = sift_detector.detectAndCompute(prev_image, None)

print(f"intrinsics:\n{camera_intrinsics}")
global_pose = np.eye(4)  # 4x4 Identity matrix
P0 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for the first camera

for iteration, (curr_image, actual_pose, image_index) in enumerate(dataset_loader):
    print(f"Processing frame {image_index}...")

    curr_keypoints, curr_descriptors = sift_detector.detectAndCompute(curr_image, None)
    matches = flann_matcher.knnMatch(prev_descriptors, curr_descriptors, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.4 * n.distance]
    pts_prev, pts_curr = extract_keypoints(prev_keypoints, curr_keypoints, good_matches)

    E, inlier_mask = cv2.findEssentialMat(
        pts_curr, pts_prev, camera_intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )

    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i, 0]]
    pts_prev_inliers, pts_curr_inliers = extract_keypoints(prev_keypoints, curr_keypoints, inlier_matches)

    _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev, camera_intrinsics)

    relative_pose = construct_homogeneous_matrix(R, t)
    global_pose = global_pose @ relative_pose

    # Update the projection matrix for the new pose and triangulate new points
    P1 = global_pose[:3]  # New projection matrix
    new_pts4D = cv2.triangulatePoints(camera_intrinsics @ P0, camera_intrinsics @ P1, pts_prev_inliers, pts_curr_inliers)
    new_pts3D = new_pts4D[:3] / new_pts4D[3]  # Convert from homogeneous to 3D coordinates

    
    inverse_pose = np.linalg.inv(global_pose)
    transformed_points_3D = (inverse_pose[:3, :3] @ new_pts3D) + inverse_pose[:3, 3:4]
    
    # Visualization and updates for next iteration
    curr_image = draw_lines_onto_image(curr_image, pts_prev, pts_curr)
    visualizer.update_image(image=curr_image)
    visualizer.update_world(pose=global_pose, points_3D=transformed_points_3D)
    visualizer.update_line_chart(None)
    visualizer.update_points_plot(pts_curr)
    visualizer.redraw()

    prev_image = curr_image.copy()
    prev_keypoints, prev_descriptors = curr_keypoints, curr_descriptors

    if not plt.get_fignums():
        break
