import cv2
import numpy as np

np.set_printoptions(precision=3, suppress=True)

from constants import *
from src.camera_calibration import *
from src.initialization import *
from src.utils import *
from src.data_loaders import *
from src.continuous_operation import *
from src.visualization import *

sift_params = {
    "nfeatures": 0,
    "nOctaveLayers": 3,
    "contrastThreshold": 0.02,
    "edgeThreshold": 50,
    "sigma": 1.6,
}
sift_detector = cv2.SIFT_create(**sift_params)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

dataset_loader = ParkingDataLoader(
    PARKING_DATA_DIR_PATH,
    init_frame_indices=[0, 2],
    image_type=cv2.IMREAD_UNCHANGED,
)

dataset_loader = KittiDataLoader(
    KITTI_DATA_DIR_PATH,
    init_frame_indices=[0, 2],
    image_type=cv2.IMREAD_UNCHANGED,
)


visualizer = VOVisualizer()

camera_intrinsics = dataset_loader.load_camera_intrinsics()
initialization_images = dataset_loader.get_initialization_frames()
prev_image = initialization_images[-1]
prev_keypoints, prev_descriptors = sift_detector.detectAndCompute(prev_image, None)

global_pose = np.eye(4)  # 4x4 Identity matrix
for iteration, (curr_image, actual_pose, image_index) in enumerate(dataset_loader):
    print(f"Processing frame {image_index}...")

    curr_keypoints, curr_descriptors = sift_detector.detectAndCompute(curr_image, None)
    matches = flann_matcher.knnMatch(prev_descriptors, curr_descriptors, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.4 * n.distance]

    # extract the matched keypoints' coordinates
    pts_prev = np.float32(
        [prev_keypoints[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    pts_curr = np.float32(
        [curr_keypoints[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    E, mask = cv2.findEssentialMat(
        pts_curr,
        pts_prev,
        camera_intrinsics,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )

    # TODO: how can we use the mask to filter out outliers?
    
    # recover the relative camera pose
    _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev, camera_intrinsics)

    estimated_pose = construct_homogeneous_matrix(R, t)
    global_pose = global_pose @ estimated_pose
    print(f"Estimated pose:\n{global_pose}")
    print(f"Actual pose:\n{actual_pose}")

    curr_image = draw_lines(
        curr_image, pts_prev, pts_curr
    )  # draw the lines between the matched keypoints
    visualizer.update_plot(
        global_pose, curr_image
    )  # show the current image and 3d visualization

    prev_image = curr_image.copy()  # save current image as previous for next iteration
    prev_keypoints, prev_descriptors = curr_keypoints, curr_descriptors

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
