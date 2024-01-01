import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils import *
from src.data_loaders import ParkingDataLoader, KittiDataLoader
from src.visualization import VOVisualizer
from src.performance_metrics import calculate_reprojection_error

# SIFT feature detector and FLANN matcher
sift_detector = cv2.SIFT_create(
    nfeatures=1000, nOctaveLayers=5, contrastThreshold=0.05, edgeThreshold=15, sigma=1.6
)

matcher = cv2.FlannBasedMatcher()
# matcher = cv2.BFMatcher()
# visualizer = VOVisualizer()

# Data loader
dataset_loader = ParkingDataLoader(init_frame_indices=[0, 2])
# dataset_loader = KittiDataLoader(init_frame_indices=[0, 2])

# Helper function to extract keypoints
extract_keypoints = lambda kpts1, kpts2, matches: (
    np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
    np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
)

pose_a_actual = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

pose_b_actual = np.array(
    [
        [1.0, 0.0, 0.0, 0.3],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpen_image = lambda img: cv2.filter2D(img, -1, sharpening_kernel)

relative_pose_ground_truth = np.linalg.inv(pose_a_actual) @ pose_b_actual
print(f"relative_pose_ground_truth:\n{relative_pose_ground_truth}")

# Load camera intrinsics and initialization images
camera_intrinsics = dataset_loader.load_camera_intrinsics()
initialization_images = dataset_loader.get_initialization_frames()
# initialization_images = [sharpen_image(img) for img in initialization_images]
print(f"intrinsics:\n{camera_intrinsics}")

keypoints_a, descriptors_a = sift_detector.detectAndCompute(
    initialization_images[0], None
)
keypoints_b, descriptors_b = sift_detector.detectAndCompute(
    initialization_images[-1], None
)

matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
print(f"Number of good matches: {len(good_matches)}")

match_img = cv2.drawMatches(
    initialization_images[0],
    keypoints_a,
    initialization_images[-1],
    keypoints_b,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
cv2.imshow("Matching", match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

pts_a, pts_b = extract_keypoints(keypoints_a, keypoints_b, good_matches)

# compute essential matrix
E, inlier_mask = cv2.findEssentialMat(
    pts_b,
    pts_a,
    camera_intrinsics,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0,
)

# TODO: get inlier matches
pts_a_inliers = pts_a[inlier_mask.ravel() == 1]
pts_b_inliers = pts_b[inlier_mask.ravel() == 1]
print(f"Number of inlier matches: {len(pts_a_inliers)}")
print(f"Number of outlier matches: {len(pts_a) - len(pts_a_inliers)}")

# (pts_a_inliers, pts_b_inliers) = (pts_b_inliers, pts_a_inliers)
# ! -> flipped pts_a and pts_b to flip coordinate system convention ?
_, R, t, mask = cv2.recoverPose(E, pts_a_inliers, pts_b_inliers, camera_intrinsics)

R = np.linalg.inv(R)
t = -R @ t

# account for scale ambiguity using ground truth of initialization frames..
t = t * 0.3
relative_pose_estimate = construct_homogeneous_matrix(R, t)
print(f"relative_pose_estimate:\n{relative_pose_estimate}")
print(f"relative_pose_ground_truth:\n{relative_pose_ground_truth}")
print(compare_poses(relative_pose_ground_truth, relative_pose_estimate))

# TRIANGULATION
P0 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for the first camera
P1 = relative_pose_estimate[:3]  # Projection matrix for the second camera

# Triangulate points
pts4D = cv2.triangulatePoints(
    camera_intrinsics @ P0, camera_intrinsics @ P1, pts_a_inliers, pts_b_inliers
)

# Convert from homogeneous to 3D coordinates
pts3D = pts4D[:3] / pts4D[3]

# Reproject points onto the image
reprojected_pts, _ = cv2.projectPoints(
    pts3D.T,
    relative_pose_estimate[:3, :3],
    relative_pose_estimate[:3, 3:4],
    camera_intrinsics,
    None,
)


#  calculate reprojection error
errors = calculate_reprojection_error(pts_b_inliers, reprojected_pts)
print("reprojection errors:")
print("max error:", np.max(errors))
print("mean error:", np.mean(errors))
print("std error:", np.std(errors))
print("median error:", np.median(errors))

print(f"3d points median of x axis: {np.median(pts3D[0, :])}")
print(f"3d points median of y axis: {np.median(pts3D[1, :])}")
print(f"3d points median of z axis: {np.median(pts3D[2, :])}")


def visualize_reprojection_onto_image(img, reprojected_pts, actual_pts, depths):
    reprojected_pts = reprojected_pts.reshape(-1, 2)
    reprojected_pts = reprojected_pts.astype(np.int32)
    reprojected_pts = reprojected_pts.reshape(-1, 1, 2)
    reprojected_img = img.copy()

    grayscale_values = (normalized_depths * 255).astype(np.uint8)
    colormap = np.stack(
        [grayscale_values] * 3, axis=-1
    )  # Replicate for 3 channels (BGR)

    for pt in reprojected_pts:
        cv2.circle(reprojected_img, tuple(pt[0]), 5, (255, 255, 255), -1)
        cv2.circle(
            reprojected_img, tuple(pt[0]), 6, (0, 0, 0), 1
        )  # Add outline to the circle
    # draw actual points onto the image
    actual_pts = actual_pts.astype(np.int32)
    actual_pts = actual_pts.reshape(-1, 1, 2)
    for pt, color in zip(actual_pts, colormap):
        bgr_tuple = tuple(int(c) for c in color)  # Convert to tuple of integers
        cv2.circle(reprojected_img, tuple(pt[0]), 2, bgr_tuple, -1)
        # outline in negative color
        cv2.circle(reprojected_img, tuple(pt[0]), 3, (0, 0, 0), 1)
    return reprojected_img


depths = pts3D[2, :]
normalized_depths = (depths - np.min(depths)) / (np.max(depths) - np.min(depths))
normalized_depths = 1 - normalized_depths

cv2.imshow(
    "Reprojected points",
    visualize_reprojection_onto_image(
        initialization_images[-1], reprojected_pts, pts_b_inliers, normalized_depths
    ),
)
cv2.waitKey(0)

# transform 3d points to world coordinates
transformed_points_3D = (
    relative_pose_estimate[:3, :3] @ pts3D
) + relative_pose_estimate[:3, 3:4]

rgb_image = cv2.cvtColor(initialization_images[-1], cv2.COLOR_BGR2RGB)
# make the top half of the image red
# rgb_image[: int(rgb_image.shape[0] / 2), :, :] = [255, 0, 0]

colors = []
for pt in pts_b_inliers:
    x, y = pt[0]
    color = rgb_image[int(y), int(x)]  # Extract color at pixel location
    colors.append(color)
colors = np.array(colors) / 255.0


# 3d plot in matplotlib of pose and points
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D World")
# generate test points to verify correct visualization
# ax.scatter(pts3D[0], pts3D[1], pts3D[2], c=colors, marker="o", s=12)

# visualize depth of points
ax.scatter(
    pts3D[0], pts3D[1], pts3D[2], c=normalized_depths, marker="o", s=12, cmap="gray"
)

ax.view_init(elev=-90, azim=-90)
# add camera pose to plot (relative_pose_estimate)
ax.scatter(0.3, 0, 0, c="red", marker="o", s=300)

# invert y axis
# ax.invert_xaxis()
# ax.invert_yaxis()
# ax.invert_zaxis()

# negate y axis points


plt.show()
import sys

sys.exit()

global_pose = np.eye(4)  # 4x4 Identity matrix
P0 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for the first camera

prev_image = initialization_images[-1]
prev_keypoints, prev_descriptors = sift_detector.detectAndCompute(prev_image, None)
for iteration, (curr_image, actual_pose, image_index) in enumerate(dataset_loader):
    print(f"Processing frame {image_index}...")

    curr_keypoints, curr_descriptors = sift_detector.detectAndCompute(curr_image, None)
    matches = matcher.knnMatch(prev_descriptors, curr_descriptors, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]
    number_of_good_matches = len(good_matches)
    pts_prev, pts_curr = extract_keypoints(prev_keypoints, curr_keypoints, good_matches)

    E, inlier_mask = cv2.findEssentialMat(
        pts_curr,
        pts_prev,
        camera_intrinsics,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )

    inlier_matches = [
        good_matches[i] for i in range(len(good_matches)) if inlier_mask[i, 0]
    ]
    pts_prev_inliers, pts_curr_inliers = extract_keypoints(
        prev_keypoints, curr_keypoints, inlier_matches
    )

    _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev, camera_intrinsics)
    R = np.linalg.inv(R)
    t = -R @ t

    relative_pose = construct_homogeneous_matrix(R, t)
    global_pose = global_pose @ relative_pose

    # Update the projection matrix for the new pose and triangulate new points
    P1 = global_pose[:3]  # New projection matrix
    new_pts4D = cv2.triangulatePoints(
        camera_intrinsics @ P0,
        camera_intrinsics @ P1,
        pts_prev_inliers,
        pts_curr_inliers,
    )
    new_pts3D = (
        new_pts4D[:3] / new_pts4D[3]
    )  # Convert from homogeneous to 3D coordinates

    # Reproject new_pts3D onto the image
    reprojected_pts, _ = cv2.projectPoints(
        new_pts3D.T, global_pose[:3, :3], global_pose[:3, 3:4], camera_intrinsics, None
    )
    # Calculate reprojection error

    reprojection_errors = calculate_reprojection_error(
        pts_curr_inliers, reprojected_pts
    )

    transformed_points_3D = (global_pose[:3, :3] @ new_pts3D) + global_pose[:3, 3:4]

    # Visualization and updates for next iteration
    curr_image = draw_lines_onto_image(curr_image, pts_prev, pts_curr)
    visualizer.update_image(image=curr_image)
    visualizer.update_world(pose=global_pose, points_3D=new_pts3D)
    visualizer.update_line_chart(
        {
            # "# of matches": (number_of_good_matches, iteration),
            "Mean Reprojection Error": (np.mean(reprojection_errors), iteration),
        }
    )
    visualizer.update_points_plot(pts_curr, reprojected_pts)
    visualizer.redraw()

    prev_image = curr_image.copy()
    prev_keypoints, prev_descriptors = curr_keypoints, curr_descriptors

    if not plt.get_fignums():
        break
