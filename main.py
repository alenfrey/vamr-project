import cv2
import numpy as np
import matplotlib.pyplot as plt
import pprint

from src.utils import *
from src.data_loaders import ParkingDataLoader, KittiDataLoader
from src.visualization import VOsualizer
from src.performance_metrics import calculate_reprojection_error
from src.config import get_config
from collections import namedtuple


dataset_loader = ParkingDataLoader(init_frame_indices=[0, 2])
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

# nameduple for storing detection results
Features = namedtuple("Features", ["keypoints", "descriptors"])


def detect_features(detector, img):
    """Detects features in an image"""
    keypoints, descriptors = detector.detectAndCompute(img, None)
    return Features(keypoints, descriptors)


features_a = detect_features(init_config["detector"], init_images[0])
features_b = detect_features(init_config["detector"], init_images[-1])


def match_features(matcher, features_a, features_b, lowe_ratio=0.8):
    """Detects and matches features between two images"""
    keypoints_a, descriptors_a = features_a
    keypoints_b, descriptors_b = features_b

    def extract_keypoints(kpts1, kpts2, matches):
        """Extracts the point coordinates from matched keypoints.
        (ugly as hell, due to opencv c++ and py mismatch)"""
        pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        return pts1, pts2

    # TODO: add k and distance to config and use it here
    matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches = [m for m, n in matches if m.distance < lowe_ratio * n.distance]
    print(f"Number of good matches: {len(good_matches)}")
    keypoints_a, keypoints_b = extract_keypoints(keypoints_a, keypoints_b, good_matches)
    return keypoints_a, keypoints_b, good_matches


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
    """Generates an image showing the matches between two images"""
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


keypoints_a, keypoints_b, good_matches = match_features(
    init_config["matcher"],
    features_a,
    features_b,
    init_config["lowe_ratio"],
    max_distance=init_config["match_max_dist"],
)

match_img = generate_match_image(
    init_images[0], init_images[-1], features_a, features_b, good_matches
)

cv2.imshow("Matching", match_img)
cv2.waitKey(0)

lines_img = draw_lines_onto_image(init_images[-1], keypoints_a, keypoints_b)
cv2.imshow("Lines", lines_img)
cv2.waitKey(0)


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


R, t, pts_a_inliers, pts_b_inliers = estimate_pose(
    keypoints_a, keypoints_b, K, **init_config["ransac"]
)
print(f"R:\n{R}")
print(f"t:\n{t}")

t = t * 0.3  # account for scale ambiguity using ground truth of initialization frames..
# R = np.linalg.inv(R)
# t = -R @ t

relative_pose_estimate = construct_homogeneous_matrix(R, t)
print(f"relative_pose_estimate:\n{relative_pose_estimate}")
print(f"relative_pose_ground_truth:\n{relative_pose_ground_truth}")
print(compare_poses(relative_pose_ground_truth, np.linalg.inv(relative_pose_estimate)))


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

    in_front_of_camera = np.logical_and(
        (P0 @ pts4D)[-1] < 0,  # Z-coordinate in camera 0's coordinate system
        (P1 @ pts4D)[-1] < 0,  # Z-coordinate in camera 1's coordinate system
    )
    pts3D = pts3D[:, in_front_of_camera]
    return pts3D


pts3D = triangulate_points(pts_a_inliers, pts_b_inliers, K, relative_pose_estimate)

# Use solvePnP for pose refinement
_, rvec, tvec, inliers = cv2.solvePnPRansac(pts3D.T, pts_b_inliers, K, None)
R_refined, _ = cv2.Rodrigues(rvec)
t_refined = tvec

if inliers is not None:
    pts3D = pts3D[:, inliers[:, 0]]
    pts_b_inliers = pts_b_inliers[inliers[:, 0]]

print(f"R_refined:\n{R_refined}")
print(f"t_refined:\n{t_refined}")
print(f"count of inliers: {len(pts_b_inliers)}")

refined_relative_pose_estimate = construct_homogeneous_matrix(R_refined, t_refined)
print(f"refined_relative_pose_estimate:\n{refined_relative_pose_estimate}")
print(
    compare_poses(
        relative_pose_ground_truth, np.linalg.inv(refined_relative_pose_estimate)
    )
)

# Reproject points onto the image
reprojected_pts, _ = cv2.projectPoints(
    pts3D,
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


def get_normalized_depths(pts3D):
    depths = pts3D[2, :]
    normalized_depths = (depths - np.min(depths)) / (np.max(depths) - np.min(depths))
    normalized_depths = 1 - normalized_depths
    return normalized_depths


normalized_depths = get_normalized_depths(pts3D)

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

colors = get_colors_from_image(rgb_image, pts_b_inliers)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D World")

# visualize depth of points
scatter = ax.scatter(
    pts3D[0],
    pts3D[1],
    pts3D[2],
    c=normalized_depths,
    marker="o",
    s=12,
    cmap="gray",
    alpha=1.0,
)

ax.view_init(elev=-90, azim=-90)  # viewpoint
ax.scatter(0.3, 0, 0, c="red", marker="o", s=300)  # pose
plt.show()

# import sys
# sys.exit()

# -<------------------->- Continuous Operation -<------------------->- #

# setup
cont_config = get_config("parking", "continuous")
visualizer = VOsualizer()
world_pose = relative_pose_estimate  # initial pose from bootstrapping
points_3d = pts3D  # initial 3d points from bootstrapping (triangulation)
print(f"world_pose:\n{world_pose}")
points_3d_world = world_pose[:3, :3] @ points_3d + world_pose[:3, 3:4]

print(f"points_3d: {points_3d}")
features_a = features_b  # features from last frame as initial features for next frame
for iteration, (curr_image, actual_pose, image_index) in enumerate(dataset_loader):
    print(f"Processing frame {image_index}...")

    # detect new features
    features_b = detect_features(cont_config["detector"], curr_image)

    # match new features with previous features
    keypoints_a, keypoints_b, good_matches = match_features(
        cont_config["matcher"],
        features_a,
        features_b,
        cont_config["lowe_ratio"],
        max_distance=cont_config["match_max_dist"],
    )

    # Estimate relative pose
    R, t, pts_a_inliers, pts_b_inliers = estimate_pose(
        keypoints_a, keypoints_b, K, **cont_config["ransac"]
    )

    # account for scale ambiguity 
    t = t * 0.1

    relative_pose = construct_homogeneous_matrix(R, t)
    print(f"relative_pose:\n{relative_pose}")
    world_pose = world_pose @ np.linalg.inv(relative_pose)
    print(f"global_pose:\n{world_pose}")

    # Triangulate points
    pts3D = triangulate_points(pts_a_inliers, pts_b_inliers, K, relative_pose)
    print(f"pts3D {pts3D.shape}")

    # refine pose using pnp ransac
    _, rvec, tvec, inliers = cv2.solvePnPRansac(pts3D.T, pts_b_inliers, K, None)
    R_refined, _ = cv2.Rodrigues(rvec)
    t_refined = tvec

    print(f"R_refined:\n{R_refined}")
    print(f"t_refined:\n{t_refined}")

    # reproject points
    reprojected_pts, _ = cv2.projectPoints(
        pts3D,
        relative_pose[:3, :3],
        relative_pose[:3, 3:4],
        K,
        None,
    )

    # Transform new 3D points to the world coordinate system
    transformed_points_3D = (world_pose[:3, :3] @ pts3D) + world_pose[:3, 3:4]
    points_3d_world = transformed_points_3D
    print(f"len(points_3d_world): {len(points_3d_world)}")

    # Visualization and updates for next iteration
    curr_image = draw_lines_onto_image(curr_image, keypoints_a, keypoints_b)

    visualizer.update_image(image=curr_image)

    visualizer.update_world(
        pose=world_pose, points_3D=points_3d_world, ground_truth_pose=actual_pose
    )

    number_of_good_matches = len(good_matches)
    visualizer.update_line_chart(
        {
            "# of matches": (number_of_good_matches, iteration),
        }
    )
    pts_curr = keypoints_b.reshape(-1, 1, 2)
    visualizer.update_points_plot(pts_curr, reprojected_pts)
    visualizer.redraw()

    # update for next iteration
    features_a = features_b
    points_3d_world = points_3d_world[:, -len(transformed_points_3D) :]

    if not plt.get_fignums():
        break
