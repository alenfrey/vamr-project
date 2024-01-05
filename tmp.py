import cv2
import numpy as np
import matplotlib.pyplot as plt
import pprint

from src.utils import *
from src.data_loaders import ParkingDataLoader, KittiDataLoader
from src.visualization import VOsualizer
from src.performance_metrics import calculate_reprojection_error
from src.config import get_config


dataset_loader = ParkingDataLoader(
    init_frame_indices=[0, 2], image_type=cv2.IMREAD_GRAYSCALE
)
dataset_name = str(dataset_loader)

# -<------------------->- Initialization -<------------------->- #

init_config = get_config(dataset_name=dataset_name, mode="initialization")
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(init_config)

K = dataset_loader.load_camera_intrinsics()
init_images, init_poses, init_indices = zip(*dataset_loader.get_initialization_data())
pose_a_actual, pose_b_actual = init_poses[0], init_poses[-1]
relative_pose_ground_truth = np.linalg.inv(pose_a_actual) @ pose_b_actual
print(f"relative_pose_ground_truth:\n{relative_pose_ground_truth}")
print(f"intrinsics:\n{K}")

img_a, img_b = init_images[0], init_images[-1]

feature_params = dict(maxCorners=300, blockSize=11)

detector = cv2.GFTTDetector_create(**feature_params)
keypoints = detector.detect(img_a)
points_a = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
print(f"points_a shape: {points_a.shape}")

lk_params = dict(
    winSize=(20, 20),
    maxLevel=1,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

points_b, status, err = cv2.calcOpticalFlowPyrLK(
    img_a, img_b, points_a, None, **lk_params
)

if points_b is not None:
    print(f"points is not None")
    points_a = points_a[status == 1]
    points_b = points_b[status == 1]
    err = err[status == 1]

print(f"mean error: {np.mean(err)}")
print(f"median error: {np.median(err)}")
print(f"max error: {np.max(err)}")
print(f"min error: {np.min(err)}")
print(f"std error: {np.std(err)}")

threshold = np.mean(err) + 2 * np.std(err)
print(f"threshold: {threshold}")
good_points_mask = err < threshold

points_a = points_a[good_points_mask]
points_b = points_b[good_points_mask]

print(f"points_a shape: {points_a.shape}")
print(f"points_b shape: {points_b.shape}")
print(f"status: {status.shape}")
print(f"err: {err.shape}")


print(f"points_a after filtering: {points_a.shape}")
print(f"points_b after filtering: {points_b.shape}")

E, mask = cv2.findEssentialMat(
    points_a, points_b, K, method=cv2.RANSAC, prob=0.999, threshold=2.0
)


# filter matches based on mask
points_a = points_a[mask.ravel() == 1]
points_b = points_b[mask.ravel() == 1]

# draw lines between the points from KLT
p1 = points_a.reshape(-1, 1, 2)
p2 = points_b.reshape(-1, 1, 2)
img_a = draw_lines_onto_image(img_a, p1, p2)
cv2.imshow("img_a", img_a)
cv2.waitKey(0)
cv2.destroyAllWindows()

_, R, t, mask = cv2.recoverPose(E, points_a, points_b, K)

print(f"R:\n{R}")
print(f"t:\n{t}")

t = t * 0.3  # account for scale ambiguity using ground truth of initialization frames..
R = np.linalg.inv(R)
t = -R @ t

print(f"R:\n{R}")
print(f"t:\n{t}")

relative_pose_estimate = construct_homogeneous_matrix(R, t)
print(f"relative_pose_estimate:\n{relative_pose_estimate}")
print(f"relative_pose_ground_truth:\n{relative_pose_ground_truth}")
print(compare_poses(relative_pose_ground_truth, np.linalg.inv(relative_pose_estimate)))


def triangulate_points_monocular(pts_a, pts_b, K, R, t):
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ np.hstack((R, t))

    # Reshape pts_a and pts_b from Nx1x2 to 2xN for triangulation
    pts_a = pts_a.reshape(-1, 2).T
    pts_b = pts_b.reshape(-1, 2).T

    # Triangulate points
    pts4D = cv2.triangulatePoints(P0, P1, pts_a, pts_b)

    # Convert from homogeneous to 3D coordinates
    pts3D = pts4D[:3] / pts4D[3]

    return pts3D, pts4D


points_3d, points_4d = triangulate_points_monocular(points_a, points_b, K, R, t)

print(f"points_3d.shape: {points_3d.shape}")


# Use solvePnP for pose refinement
_, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d.T, points_b, K, None)
R_refined, _ = cv2.Rodrigues(rvec)
t_refined = tvec


refined_relative_pose_estimate = construct_homogeneous_matrix(R_refined, t_refined)
print(f"refined_relative_pose_estimate:\n{refined_relative_pose_estimate}")
print(
    compare_poses(
        relative_pose_ground_truth, np.linalg.inv(refined_relative_pose_estimate)
    )
)

# Reproject points onto the image
reprojected_pts, _ = cv2.projectPoints(
    points_3d,
    relative_pose_estimate[:3, :3],
    relative_pose_estimate[:3, 3:4],
    K,
    None,
)

#  calculate reprojection error
errors = calculate_reprojection_error(points_b, reprojected_pts)
print("reprojection errors:")
print("max error:", np.max(errors))
print("mean error:", np.mean(errors))
print("std error:", np.std(errors))
print("median error:", np.median(errors))


print(f"3d points median of x axis: {np.median(points_3d[0, :])}")
print(f"3d points median of y axis: {np.median(points_3d[1, :])}")
print(f"3d points median of z axis: {np.median(points_3d[2, :])}")


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


def get_normalized_depths(pts3D):
    depths = pts3D[2, :]
    normalized_depths = (depths - np.min(depths)) / (np.max(depths) - np.min(depths))
    normalized_depths = 1 - normalized_depths
    return normalized_depths


normalized_depths = get_normalized_depths(points_3d)

cv2.imshow(
    "Reprojected points",
    visualize_reprojection_onto_image(
        init_images[-1], reprojected_pts, points_b, normalized_depths
    ),
)
cv2.waitKey(0)

# transform 3d points to world coordinates
transformed_points_3D = (
    relative_pose_estimate[:3, :3] @ points_3d
) + relative_pose_estimate[:3, 3:4]

rgb_image = cv2.cvtColor(init_images[-1], cv2.COLOR_BGR2RGB)
# make the top half of the image red
# rgb_image[: int(rgb_image.shape[0] / 2), :, :] = [255, 0, 0]

# colors = get_colors_from_image(rgb_image, points_b.reshape(-1, 2))

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D World")

# visualize depth of points
scatter = ax.scatter(
    points_3d[0],
    points_3d[1],
    points_3d[2],
    c=normalized_depths,
    marker="o",
    s=12,
    cmap="gray",
    alpha=1.0,
)

ax.view_init(elev=-90, azim=-90)  # viewpoint
ax.scatter(0.3, 0, 0, c="red", marker="o", s=300)  # pose
plt.show()

import sys

sys.exit()

# -<------------------->- Continuous Operation -<------------------->- #

# setup
cont_config = get_config("parking", "continuous")
visualizer = VOsualizer()
world_pose = relative_pose_estimate
points_3d_world = pts3D
print(f"world_pose:\n{world_pose}")

features_a = features_b
points_3d = pts3D  # 3d points from the initialization
for iteration, (curr_image, actual_pose, image_index) in enumerate(dataset_loader):
    print(f"Processing frame {image_index}...")

    features_b = detect_features(cont_config["detector"], curr_image)

    keypoints_a, keypoints_b, good_matches = match_features(
        cont_config["matcher"],
        features_a,
        features_b,
        cont_config["lowe_ratio"],
        max_distance=cont_config["match_max_dist"],
    )

    # Estimate pose using essential mat (2d)
    R, t, pts_a_inliers, pts_b_inliers = estimate_pose(
        keypoints_a, keypoints_b, K, **cont_config["ransac"]
    )

    # get correspondences 2d-3d and get pose using pnp

    t = t * 0.3

    relative_pose = construct_homogeneous_matrix(R, t)
    print(f"relative_pose:\n{relative_pose}")
    world_pose = world_pose @ np.linalg.inv(relative_pose)
    print(f"global_pose:\n{world_pose}")

    # Triangulate points
    pts3D, pts4D = triangulate_points(pts_a_inliers, pts_b_inliers, K, relative_pose)

    # Transform new 3D points to the world coordinate system
    transformed_points_3D = (world_pose[:3, :3] @ pts3D) + world_pose[:3, 3:4]

    # Combine with existing world points
    points_3d_world = np.hstack((points_3d_world, transformed_points_3D))
    print(f"len(points_3d_world): {len(points_3d_world)}")
    # Visualization and updates for next iteration
    curr_image = draw_lines_onto_image(curr_image, keypoints_a, keypoints_b)

    visualizer.update_image(image=curr_image)

    visualizer.update_world(pose=world_pose, points_3D=points_3d_world)

    visualizer.redraw()

    # update for next iteration
    features_a = features_b
    points_3d_world = points_3d_world[:, -len(transformed_points_3D) :]

    if not plt.get_fignums():
        break
