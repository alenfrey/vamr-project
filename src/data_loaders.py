"""
This file contains the data loaders for the VO pipeline, to load different datasets.
"""

from pathlib import Path
import numpy as np
import cv2


class VODataLoader:
    def __init__(self, dataset_path, init_frame_indices=None):
        self.dataset_path = Path(dataset_path)
        self.image_paths = self.setup_image_loader()
        self.poses = self.load_poses()
        self.init_frame_indices = init_frame_indices or []
        self.current_image_index = 0

    def load_camera_intrinsics(self):
        pass

    def setup_image_loader(self):
        pass

    def load_poses(self):
        pass

    def get_initialization_frames(self):
        return [
            self.load_image(self.image_paths[idx]) for idx in self.init_frame_indices
        ]

    def __iter__(self):
        # start iterating from the frame immediately after the last initialization frame
        self.current_image_index = (
            max(self.init_frame_indices) + 1 if self.init_frame_indices else 0
        )
        return self

    def __next__(self):
        if self.current_image_index >= len(self.image_paths):
            raise StopIteration

        image = self.load_image(self.image_paths[self.current_image_index])
        pose = self.poses[self.current_image_index] if self.poses is not None else None
        index = self.current_image_index  # for returning the image index
        self.current_image_index += 1

        return image, pose, index

    def load_image(self, image_path):
        pass


class ParkingDataLoader(VODataLoader):
    def __init__(
        self, dataset_path, init_frame_indices=None, image_type=cv2.IMREAD_GRAYSCALE
    ):
        super().__init__(dataset_path, init_frame_indices)
        # image_type can be cv2.IMREAD_GRAYSCALE or cv2.IMREAD_COLOR or cv2.IMREAD_UNCHANGED
        self.image_type = image_type

    def load_camera_intrinsics(self):
        intrinsic_path = self.dataset_path / "K.txt"
        K = np.genfromtxt(intrinsic_path, delimiter=",", dtype=float, usecols=(0, 1, 2))
        return K

    def setup_image_loader(self):
        image_directory = self.dataset_path / "images"
        return sorted(image_directory.glob("*.png"))

    def load_image(self, image_path):
        return cv2.imread(str(image_path), self.image_type)

    def load_poses(self):
        poses_path = self.dataset_path / "poses.txt"
        flat_poses = np.genfromtxt(poses_path, dtype=float)
        # convert flat poses to 4x4 matrices, TODO: maybe we need to use the method
        # from the exercise sheet instead of this one
        poses = [np.vstack((pose.reshape(3, 4), [0, 0, 0, 1])) for pose in flat_poses]
        return np.array(poses)
