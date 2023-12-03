from constants import *
from src.camera_calibration import *
from src.initialization import *
from src.utils import *
from src.data_loaders import *

dataset_loader = ParkingDataLoader(PARKING_DATA_DIR_PATH)

# example of loading the parking dataset with specific initialization frames
init_frames = [0, 2]  # e.g., frame 1 and frame 3 for initialization
dataset_loader = ParkingDataLoader(
    PARKING_DATA_DIR_PATH, init_frame_indices=init_frames
)

camera_intrinsics = dataset_loader.load_camera_intrinsics()
initialization_images = dataset_loader.get_initialization_frames()

# Main VO Loop, streamed from the dataset
dataset_loader.current_image_index = len(
    initialization_images
)  # skip initialization frames

for idx, (image, pose) in enumerate(dataset_loader):
    # TODO: implement the main VO loop here

    cv2.imshow("Image Stream", image)

    # Wait for 30ms or until a key is pressed; if 'q' is pressed, break the loop
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# When everything is done, release the OpenCV window
cv2.destroyAllWindows()
