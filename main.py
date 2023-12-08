import cv2

from constants import *
from src.camera_calibration import *
from src.initialization import initialize_vo, ensure_grayscale, initialize_keypoints_harris, initialize_keypoints_sift
from src.utils import *
from src.data_loaders import ParkingDataLoader
from src.continuous_operation import (
    associate_keypoints,
    estimate_pose,
    triangulate_landmarks,
    update_map_and_state,
    visualize_vo,
)
from src.performance_metrics import FPSCounter

dataset_loader = ParkingDataLoader(PARKING_DATA_DIR_PATH)
fps_counter = FPSCounter()

# example of loading the parking dataset with specific initialization frames
dataset_loader = ParkingDataLoader(
    PARKING_DATA_DIR_PATH,
    init_frame_indices=[0, 2],  # (frame 1 and frame 3 for initialization)
    image_type=cv2.IMREAD_GRAYSCALE,  # can also use cv2.IMREAD_COLOR for color images
)

camera_intrinsics = dataset_loader.load_camera_intrinsics()
initialization_images = dataset_loader.get_initialization_frames()
R, t = initialize_vo(initialization_images, camera_intrinsics)
prev_image = initialization_images[-1]
# initialize keypoints
# image_1, keypoints_1, descriptors_1 = initialize_keypoints(prev_image)
# initialize parameter for Harris (TODO: tune)
corner_patch_size = 9
kernel_size = 3
harris_kappa = 0.08
num_keypoints = 200
nonmaximum_supression_radius = 8
descriptor_radius = 9
match_lambda = 4

# run Harris on first frame
keypoints_1 = initialize_keypoints_harris(prev_image, corner_patch_size, kernel_size, harris_kappa)

prev_keypoints = None

for iteration, (curr_image, pose, image_index) in enumerate(dataset_loader):
    print(f"Processing frame {image_index}...")
    # TODO: implement the main VO loop here, by implementing functions in src/continuous_operation.py or
    # similar modules in src/ directory.

    # Keypoint Association
    # Use KLT or another method to find keypoints in the current frame and associate them with previous frame's keypoints.
    # if prev_keypoints is None:
    #     prev_keypoints = keypoints_1
    associate_keypoints(curr_image, prev_image, prev_keypoints)

    # Pose Estimation
    # Use the associated keypoints to estimate the current camera pose.
    estimate_pose(..., ..., ..., ...)

    # Landmark Triangulation
    # Triangulate new landmarks based on current and previous frames.
    triangulate_landmarks(..., ..., ...)

    # Map and State Update
    # Update your map and system state with new information.
    update_map_and_state(..., ..., ..., ...)

    # Visualization
    # Visualize keypoints, camera trajectory, and landmarks.
    visualize_vo(curr_image, ..., ..., ...)

    # Performance Evaluation
    # Calculate Performance Metrics
    # Update FPS counter and display FPS
    fps = fps_counter.update()
    fps_counter.put_fps_on_image(curr_image, fps)

    cv2.imshow("Image Stream", curr_image)

    # Wait for 30ms or until a key is pressed; if 'q' is pressed, break the loop
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# When everything is done, release the OpenCV window
cv2.destroyAllWindows()
