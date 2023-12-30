import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_reprojection_error(projected_points, actual_points):
    reprojection_errors = np.linalg.norm(projected_points - actual_points, axis=2)
    mean_error = np.mean(reprojection_errors)
    std_error = np.std(reprojection_errors)
    return reprojection_errors, mean_error, std_error


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
