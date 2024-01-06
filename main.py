import cv2
import numpy as np
import pprint
import matplotlib.pyplot as plt

from src.utils import *
from src.data_loaders import ParkingDataLoader, KittiDataLoader, MalagaDataLoader
from src.visualization import VOsualizer, scatter_3d_points
from src.performance_metrics import calculate_reprojection_error, FPSCounter
from src.config import get_config
from src.visual_odometry import *
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

features_a = detect_features(init_config["detector"], init_images[0])
features_b = detect_features(init_config["detector"], init_images[-1])

keypoints_a, keypoints_b, good_matches = match_features(
    init_config["matcher"],
    features_a,
    features_b,
    init_config["lowe_ratio"],
    max_distance=init_config["match_max_dist"],
)

# Generate image of keypoint matching and show it
match_img = generate_match_image(
    init_images[0], init_images[-1], features_a, features_b, good_matches
)
show_image(match_img, "Matches")

# Draw lines on image based on keypoint matching an show it
lines_img = draw_lines_onto_image(init_images[-1], keypoints_a, keypoints_b)
show_image(lines_img, "Lines")

# Estimate pose using essential matrix
R, t, pts_a_inliers, pts_b_inliers = estimate_pose(
    keypoints_a, keypoints_b, K, **init_config["ransac"]
)

print(f"R:\n{R}")
print(f"t:\n{t}")

# account for scale ambiguity using ground truth of initialization frames..
t = t * init_config["translation_scale"]

relative_pose_estimate = construct_homogeneous_matrix(R, t)
print(f"relative_pose_estimate:\n{relative_pose_estimate}")
print(f"relative_pose_ground_truth:\n{relative_pose_ground_truth}")
print(compare_poses(relative_pose_ground_truth, np.linalg.inv(relative_pose_estimate)))

points_3d = triangulate_points(pts_a_inliers, pts_b_inliers, K, relative_pose_estimate)

# print cound of 3d points
print(f"number of 3d points: {len(points_3d.T)}")
print(f"number of inliers: {len(pts_b_inliers)}")


_, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d.T, pts_b_inliers, K, None)
R_refined, _ = cv2.Rodrigues(rvec)
t_refined = tvec

# if inliers is not None:
#     points_3d = points_3d[:, inliers[:, 0]]
#     pts_b_inliers = pts_b_inliers[inliers[:, 0]]

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

reprojected_points = reproject_points(points_3d, R, t, K)
reprojection_errors = calculate_reprojection_error(pts_b_inliers, reprojected_points)

print(get_array_stats(reprojection_errors, "reprojection_errors"))
print(get_array_stats(points_3d[0, :], "x"))
print(get_array_stats(points_3d[1, :], "y"))
print(get_array_stats(points_3d[2, :], "z"))

normalized_depths = get_normalized_depths(points_3d)

reprojection_img = generate_reprojection_image(
    init_images[-1], reprojected_points, pts_b_inliers, normalized_depths
)

show_image(reprojection_img, "Reprojected points")

points_3d_world = to_world_coordinates(points_3d, pose=relative_pose_estimate)

rgb_image = cv2.cvtColor(init_images[-1], cv2.COLOR_BGR2RGB)
colors = get_colors_from_image(rgb_image, pts_b_inliers)

scatter_3d_points(
    points_3d_world, colors=normalized_depths, title="3D World from Initialization"
)

# -<------------------->- Continuous Operation -<------------------->- #

# setup
fps_counter = FPSCounter()
cont_config = get_config("parking", "continuous")
visualizer = VOsualizer()
world_pose = relative_pose_estimate  # initial pose from bootstrapping
points_3d = points_3d  # initial 3d points from bootstrapping (triangulation)
print(f"world_pose:\n{world_pose}")
points_3d_world = to_world_coordinates(points_3d, pose=world_pose)

print(f"points_3d_world: {points_3d_world}")
print(f"points_3d_world.shape: {points_3d_world.shape}")

print(f"points_3d: {points_3d}")
print(f"number of 3d points: {len(points_3d.T)}")
all_points = set()
for point in points_3d_world.T:
    # Convert the point to a tuple and add it to the set
    all_points.add(tuple(point))

features_a = (
    features_b  # features from last frame as initial features qqqfor next frame
)
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

    # estimate relative pose
    R, t, pts_a_inliers, pts_b_inliers = estimate_pose(
        keypoints_a, keypoints_b, K, **cont_config["ransac"]
    )

    # account for scale ambiguity
    t = t * cont_config["translation_scale"]

    relative_pose = construct_homogeneous_matrix(R, t)
    print(f"relative_pose:\n{relative_pose}")
    world_pose = world_pose @ np.linalg.inv(relative_pose)
    print(f"global_pose:\n{world_pose}")
    print(f"number of 3d points: {len(points_3d)}")

    # triangulate points
    pts3D = triangulate_points(pts_a_inliers, pts_b_inliers, K, relative_pose)

    # compare length of points and inliers b
    print(f"len(pts3D): {len(pts3D.T)}")
    print(f"len(pts_b_inliers): {len(pts_b_inliers)}")

    # transform new 3D points to the world coordinate system
    transformed_points_3D = (world_pose[:3, :3] @ pts3D) + world_pose[:3, 3:4]
    points_3d_world = transformed_points_3D
    print(f"len(points_3d_world): {len(points_3d_world)}")

    # add new points to the list of all points
    for point in transformed_points_3D.T:
        all_points.add(tuple(point))

    rgb_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
    colors = get_colors_from_image(rgb_image, pts_b_inliers)
    # visualization and updates for next iteration
    curr_image = draw_lines_onto_image(curr_image, keypoints_a, keypoints_b)

    fps = fps_counter.update()
    fps_counter.put_fps_on_image(curr_image, fps)
    visualizer.update_image(image=curr_image)

    visualizer.update_world(
        pose=world_pose,
        points_3D=points_3d_world,
        ground_truth_pose=actual_pose,
        colors=colors,
        all_points=all_points,
    )

    number_of_good_matches = len(good_matches)
    visualizer.update_line_chart(
        {
            "# of matches": (number_of_good_matches, iteration),
        }
    )
    visualizer.redraw()

    # update for next iteration
    features_a = features_b

    if not plt.get_fignums():
        break
