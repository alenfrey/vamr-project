import cv2
import numpy as np

np.set_printoptions(precision=3, suppress=True)

from constants import *
from src.camera_calibration import *
from src.initialization import *
from src.utils import *
from src.data_loaders import ParkingDataLoader
from src.continuous_operation import *
from src.visualization import *
from src.performance_metrics import calculate_pose_error

# Initialize the SIFT detector
sift_params = {
    "nfeatures": 0,
    "nOctaveLayers": 3,
    "contrastThreshold": 0.02,  # default 0.04
    "edgeThreshold": 50,  #
    "sigma": 1.6,
}
sift_detector = cv2.SIFT_create(**sift_params)

# Initialize FLANN based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass an empty dictionary
flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

dataset_loader = ParkingDataLoader(
    PARKING_DATA_DIR_PATH,
    init_frame_indices=[0, 2],
    image_type=cv2.IMREAD_GRAYSCALE,
)

visualizer = RealTimePoseVisualizer()

camera_intrinsics = dataset_loader.load_camera_intrinsics()
initialization_images = dataset_loader.get_initialization_frames()

prev_image = initialization_images[-1]
prev_keypoints, prev_descriptors = sift_detector.detectAndCompute(prev_image, None)

global_pose = np.eye(4)  # 4x4 Identity matrix

for iteration, (curr_image, actual_pose, image_index) in enumerate(dataset_loader):
    print(f"Processing frame {image_index}...")

    # Detect keypoints and compute descriptors in the current image
    curr_keypoints, curr_descriptors = sift_detector.detectAndCompute(curr_image, None)

    # Match descriptors with the previous frame
    matches = flann_matcher.knnMatch(prev_descriptors, curr_descriptors, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    # Extract the matched keypoints' coordinates
    pts_prev = np.float32(
        [prev_keypoints[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    pts_curr = np.float32(
        [curr_keypoints[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # Compute the Essential Matrix
    E, mask = cv2.findEssentialMat(
        pts_curr,
        pts_prev,
        camera_intrinsics,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )

    # Recover the relative camera pose
    _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev, camera_intrinsics)

    estimate_pose = construct_homogeneous_matrix(R, t)
    global_pose = np.dot(global_pose, estimate_pose)
    print(f"Estimated pose:\n{global_pose}")
    
    visualizer.update_plot(global_pose, np.zeros((3, 0)))
    print(f"Actual pose:\n{actual_pose}")
    # Visualize matches
    matched_image = cv2.drawMatches(
        prev_image,
        prev_keypoints,
        curr_image,
        curr_keypoints,
        good_matches,
        None,
        matchesMask=mask.ravel().tolist(),
    )

    cv2.imshow("Matches", matched_image)

    # Update previous frame keypoints and descriptors
    prev_image = curr_image.copy()
    prev_keypoints, prev_descriptors = curr_keypoints, curr_descriptors


    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# Release the OpenCV window
cv2.destroyAllWindows()
