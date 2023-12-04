import cv2

from constants import *
from src.camera_calibration import *
from src.initialization import initialize_vo, ensure_grayscale
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
init_frames = [0, 2]  # (frame 1 and frame 3 for initialization)
dataset_loader = ParkingDataLoader(
    PARKING_DATA_DIR_PATH,
    init_frame_indices=init_frames,
    image_type=cv2.IMREAD_GRAYSCALE,  # can also use cv2.IMREAD_COLOR for color images
)

camera_intrinsics = dataset_loader.load_camera_intrinsics()
initialization_images = dataset_loader.get_initialization_frames()
R, t = initialize_vo(initialization_images, camera_intrinsics)
prev_image = initialization_images[-1]
# TODO: calculate keypoints of prev_image to use in first main loop iteration

for iteration, (curr_image, pose, image_index) in enumerate(dataset_loader):
    print(f"Processing frame {image_index}...")
    # TODO: implement the main VO loop here, by implementing functions in src/continuous_operation.py or
    # similar modules in src/ directory.

    # Keypoint Association
    # Use KLT or another method to find keypoints in the current frame and associate them with previous frame's keypoints.
    associate_keypoints(curr_image, prev_image, ...)

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
