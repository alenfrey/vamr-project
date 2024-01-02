"""
This file contains the data loaders for the VO pipeline, to load different datasets.
"""

from pathlib import Path
from constants import *
import numpy as np
import cv2

# TODO Add option to scale images and K by a factor given as an argument


class VODataLoader:
    def __init__(self, dataset_path, init_frame_indices=None):
        self.dataset_path = Path(dataset_path)
        self.image_paths = self.setup_image_loader()
        self.poses = self.load_poses()
        self.init_frame_indices = init_frame_indices or []
        self.current_image_index = 0
        self.name = self.dataset_path.name

    def load_camera_intrinsics(self):
        pass

    def setup_image_loader(self):
        pass

    def load_poses(self):
        pass

    def get_initialization_data(self):
        print(f"Total images available: {len(self.image_paths)}")  # Debugging line
        print(f"Initialization indices: {self.init_frame_indices}")  # Debugging line
        init_data = []
        for idx in self.init_frame_indices:
            image = self.load_image(self.image_paths[idx])
            pose = self.poses[idx] if self.poses is not None else None
            init_data.append((image, pose, idx))
        return init_data

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

    def __str__(self):
        return self.name

    def load_image(self, image_path):
        pass


class ParkingDataLoader(VODataLoader):
    def __init__(
        self,
        dataset_path=PARKING_DATA_DIR_PATH,
        init_frame_indices=None,
        image_type=cv2.IMREAD_UNCHANGED,
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


class KittiDataLoader(VODataLoader):
    def __init__(
        self,
        dataset_path=KITTI_DATA_DIR_PATH,
        init_frame_indices=None,
        image_type=cv2.IMREAD_UNCHANGED,
    ):
        super().__init__(dataset_path, init_frame_indices)
        self.image_type = image_type

    def load_camera_intrinsics(self):
        # Assuming we are using P2 (3x4 projection matrix for the left color camera)
        calib_path = self.dataset_path / "05" / "calib.txt"
        with open(calib_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("P0"):
                    values = line.split()[1:]
                    intrinsic_matrix = np.array([float(v) for v in values]).reshape(
                        3, 4
                    )
                    return intrinsic_matrix[:3, :3]
        return None

    def setup_image_loader(self):
        image_directory = self.dataset_path / "05" / "image_0"
        print(f"Image directory: {image_directory}")
        return sorted(image_directory.glob("*.png"))

    def load_poses(self):
        poses_path = self.dataset_path / "poses" / "05.txt"
        flat_poses = np.genfromtxt(poses_path, dtype=float)
        poses = [np.vstack((pose.reshape(3, 4), [0, 0, 0, 1])) for pose in flat_poses]
        return np.array(poses)

    def load_image(self, image_path):
        return cv2.imread(str(image_path), self.image_type)


class MalagaDataLoader(VODataLoader):
    def __init__(
        self,
        dataset_path=MALAGA_DATA_DIR_PATH,
        init_frame_indices=None,
        image_type=cv2.IMREAD_UNCHANGED,
    ):
        super().__init__(dataset_path, init_frame_indices)
        self.image_type = image_type
        self.K = np.array(
            [[837.619011, 0, 522.434637], [0, 839.808333, 402.367400], [0, 0, 1]]
        )

    def load_camera_intrinsics(self):
        return self.K

    def setup_image_loader(self):
        image_directory = self.dataset_path / "Images"
        all_images = sorted(image_directory.glob("*.jpg"))
        left_images = [img for img in all_images if "left" in img.name]
        return left_images

    def load_image(self, image_path):
        return cv2.imread(str(image_path), self.image_type)

    # load_poses method is not implemented yet, unsure of how to do it atm, need to research more
