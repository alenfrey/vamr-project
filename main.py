import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils import *
from src.data_loaders import ParkingDataLoader, KittiDataLoader
from src.visualization import VOVisualizer
from src.performance_metrics import calculate_reprojection_error
from src.config import get_config

# TODO move these to config and use them from there
sift_detector = cv2.SIFT_create(
    nfeatures=2000, nOctaveLayers=5, contrastThreshold=0.05, edgeThreshold=15, sigma=1.6
)
matcher = cv2.FlannBasedMatcher()
# TODO END

# Data loader
dataset_loader = ParkingDataLoader(init_frame_indices=[0, 2])

K = dataset_loader.load_camera_intrinsics()
init_images, init_poses, init_indices = zip(*dataset_loader.get_initialization_data())
pose_a_actual, pose_b_actual = init_poses[0], init_poses[-1]
relative_pose_ground_truth = np.linalg.inv(pose_a_actual) @ pose_b_actual
print(f"relative_pose_ground_truth:\n{relative_pose_ground_truth}")
print(f"intrinsics:\n{K}")


def detect_and_match_features(detector, matcher, img_a, img_b):
    """Detects and matches features between two images"""

    def extract_keypoints(kpts1, kpts2, matches):
        """Extracts the point coordinates from matched keypoints.
        (ugly as hell, due to opencv c++ and py mismatch)"""
        pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        return pts1, pts2

    def generate_match_image(img_a, img_b, keypoints_a, keypoints_b, good_matches):
        """Generates an image showing the matches between two images"""
        match_img = cv2.drawMatches(
            img_a,
            keypoints_a,
            img_b,
            keypoints_b,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return match_img

    keypoints_a, descriptors_a = detector.detectAndCompute(img_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(img_b, None)
    # TODO: add k and distance to config and use it here
    matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
    print(f"Number of good matches: {len(good_matches)}")
    match_img = generate_match_image(
        init_images[0], init_images[-1], keypoints_a, keypoints_b, good_matches
    )
    cv2.imshow("Matching", match_img)
    cv2.waitKey(0)

    keypoints_a, keypoints_b = extract_keypoints(keypoints_a, keypoints_b, good_matches)
    return keypoints_a, keypoints_b, good_matches


keypoints_a, keypoints_b, good_matches = detect_and_match_features(
    sift_detector, matcher, init_images[0], init_images[-1]
)

lines_img = draw_lines_onto_image(init_images[-1], keypoints_a, keypoints_b)
cv2.imshow("Lines", lines_img)
cv2.waitKey(0)


def estimate_pose(keypoints_a, keypoints_b, K):
    """Estimates the pose between two images using RANSAC"""
    E, inlier_mask = cv2.findEssentialMat(
        keypoints_a,
        keypoints_b,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
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


R, t, pts_a_inliers, pts_b_inliers = estimate_pose(keypoints_a, keypoints_b, K)
print(f"R:\n{R}")
print(f"t:\n{t}")

t = t * 0.3  # account for scale ambiguity using ground truth of initialization frames..
# R = np.linalg.inv(R)
# t = -R @ t

relative_pose_estimate = construct_homogeneous_matrix(R, t)
print(f"relative_pose_estimate:\n{relative_pose_estimate}")
print(f"relative_pose_ground_truth:\n{relative_pose_ground_truth}")
print(compare_poses(relative_pose_ground_truth, relative_pose_estimate))

def triangulate_points(pts_a, pts_b, K, relative_pose):
    """Triangulates points from two images"""
    R = relative_pose[:3, :3]  # Rotation matrix from the relative pose
    t = relative_pose[:3, 3:4]  # Translation vector from the relative pose
    P0 = np.hstack(
        (np.eye(3), np.zeros((3, 1)))
    )  # Projection matrix for the first camera
    P1 = np.hstack((R, t))  # Projection matrix for the second camera
    P0 = K @ P0  # Apply the intrinsic matrix to the first camera
    P1 = K @ P1  # Apply the intrinsic matrix to the second camera
    pts4D = cv2.triangulatePoints(P0, P1, pts_a, pts_b)
    pts3D = pts4D[:3] / pts4D[3]  # Convert from homogeneous to 3D coordinates
    return pts3D, pts4D


pts3D, pts4D = triangulate_points(pts_a_inliers, pts_b_inliers, K, relative_pose_estimate)




# Reproject points onto the image
reprojected_pts, _ = cv2.projectPoints(
    pts3D.T,
    relative_pose_estimate[:3, :3],
    relative_pose_estimate[:3, 3:4],
    K,
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

    grayscale_values = (depths * 255).astype(np.uint8)
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
        init_images[-1], reprojected_pts, pts_b_inliers, normalized_depths
    ),
)
cv2.waitKey(0)

# transform 3d points to world coordinates
transformed_points_3D = (
    relative_pose_estimate[:3, :3] @ pts3D
) + relative_pose_estimate[:3, 3:4]

rgb_image = cv2.cvtColor(init_images[-1], cv2.COLOR_BGR2RGB)
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

# visualize depth of points
ax.scatter(
    pts3D[0],
    pts3D[1] * 0.5,
    pts3D[2],
    c=normalized_depths,
    marker="o",
    s=12,
    cmap="gray",
    alpha=1.0,
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

prev_image = init_images[-1]
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
        K,
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

    _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev, K)
    R = np.linalg.inv(R)
    t = -R @ t

    relative_pose = construct_homogeneous_matrix(R, t)
    global_pose = global_pose @ relative_pose

    # Update the projection matrix for the new pose and triangulate new points
    P1 = global_pose[:3]  # New projection matrix
    new_pts4D = cv2.triangulatePoints(
        K @ P0,
        K @ P1,
        pts_prev_inliers,
        pts_curr_inliers,
    )
    new_pts3D = (
        new_pts4D[:3] / new_pts4D[3]
    )  # Convert from homogeneous to 3D coordinates

    # Reproject new_pts3D onto the image
    reprojected_pts, _ = cv2.projectPoints(
        new_pts3D.T, global_pose[:3, :3], global_pose[:3, 3:4], K, None
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
