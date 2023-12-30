import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_reprojection_error(projected_points_2D, original_keypoints):
    """
    Calculate the average reprojection error.

    Parameters:
    projected_points_2D (np.array): Projected 2D points from 3D points.
    original_keypoints (np.array): Original 2D keypoints used for triangulation.

    Returns:
    float: Average reprojection error.
    """
    total_error = 0
    count = 0

    for projected, original in zip(projected_points_2D, original_keypoints):
        # Reshape projected point for compatibility
        projected_reshaped = projected[0].ravel()

        # Ensure both points are not NaN and have the same dimension
        if (
            not np.isnan(projected_reshaped).any()
            and not np.isnan(original.ravel()).any()
            and len(projected_reshaped) == len(original.ravel())
        ):
            error = cv2.norm(original, projected_reshaped, cv2.NORM_L2)
            total_error += error
            count += 1

    if count > 0:
        average_error = total_error / count
    else:
        average_error = float("inf")

    return average_error


class FPSCounter:
    def __init__(self):
        self.prev_frame_time = time.time()
        self.curr_frame_time = time.time()

    def update(self):
        self.curr_frame_time = time.time()
        fps = 1 / (self.curr_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.curr_frame_time
        return fps

    def put_fps_on_image(
        self,
        image,
        fps,
        position=(10, 30),
        font_scale=1,
        main_color=(100, 255, 0),
        outline_color=(0, 0, 0),
        thickness=2,
    ):
        fps_text = f"FPS: {fps:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Draw the outline
        cv2.putText(
            image,
            fps_text,
            position,
            font,
            font_scale,
            outline_color,
            thickness + 2,
            cv2.LINE_AA,
        )
        # Draw the main text
        cv2.putText(
            image,
            fps_text,
            position,
            font,
            font_scale,
            main_color,
            thickness,
            cv2.LINE_AA,
        )
