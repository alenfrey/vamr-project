import cv2
import numpy as np
import pytest

from pathlib import Path
from hypothesis import given, strategies as st
from src.data_loaders import ParkingDataLoader
from constants import PARKING_DATA_DIR_PATH

# PARKING DATA LOADER TESTS


@given(
    init_frames=st.lists(st.integers(min_value=0, max_value=10), min_size=2, max_size=5)
)
def test_random_initialization_frames(init_frames):
    loader = ParkingDataLoader(PARKING_DATA_DIR_PATH, init_frame_indices=init_frames)
    # test if loader behaves as expected with these random init_frames


@given(
    image_type=st.sampled_from(
        [cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED]
    )
)
def test_different_image_types(image_type):
    loader = ParkingDataLoader(PARKING_DATA_DIR_PATH, image_type=image_type)
    # test if loader handles different image types correctly


def test_iteration_start():
    """Test that iteration starts at the correct frame after the last initialization frame."""
    init_frames = [0, 2]  # Example initialization frames
    loader = ParkingDataLoader(PARKING_DATA_DIR_PATH, init_frame_indices=init_frames)
    first_iterated_frame_index = next(iter(loader))[
        2
    ]  # Assuming the third element is the frame index
    assert first_iterated_frame_index == max(init_frames) + 1


def test_unordered_init_frames():
    """Test loader behavior with unordered initialization frame indices."""
    init_frames = [3, 1]  # Unordered initialization frames
    loader = ParkingDataLoader(PARKING_DATA_DIR_PATH, init_frame_indices=init_frames)
    first_iterated_frame_index = next(iter(loader))[2]
    assert (
        first_iterated_frame_index == max(init_frames) + 1
    ), "Should start after the highest init frame index."


def test_data_loader_construction():
    loader = ParkingDataLoader(PARKING_DATA_DIR_PATH, init_frame_indices=[0, 2])
    assert isinstance(loader, ParkingDataLoader)
    assert loader.dataset_path == Path(PARKING_DATA_DIR_PATH)
    assert loader.init_frame_indices == [0, 2]


def test_load_camera_intrinsics():
    """Test if camera intrinsics are loaded correctly and have the expected shape."""
    loader = ParkingDataLoader(PARKING_DATA_DIR_PATH)
    K = loader.load_camera_intrinsics()
    assert isinstance(K, np.ndarray), "Camera intrinsics should be a NumPy array."
    assert K.shape == (3, 3), "Camera intrinsics matrix should be 3x3."


def test_invalid_dataset_path():
    """Test the loader's response to an invalid dataset path."""
    with pytest.raises(
        Exception
    ):  # Replace Exception with a more specific exception if your code raises one.
        invalid_loader = ParkingDataLoader("invalid/path")
        invalid_loader.setup_image_loader()  # Assuming this method would fail with an invalid path.


def test_load_image():
    loader = ParkingDataLoader(PARKING_DATA_DIR_PATH)
    image_path = next(iter(loader.image_paths))
    image = loader.load_image(image_path)
    assert isinstance(image, np.ndarray)  # assuming images are loaded as numpy arrays
