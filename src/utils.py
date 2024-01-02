"""
This file contains utility functions that are used in the other files.
"""

import requests
import zipfile
import io
import time
import numpy as np
import cv2
from tqdm import tqdm
from functools import wraps
import cv2
import cv2
import numpy as np


def get_colors_from_image(image, points):
    colors = []
    for pt in points:
        x, y = pt[0]
        color = image[int(y), int(x)]  # extract color at pixel location
        colors.append(color)
    colors = np.array(colors) / 255.0
    return colors


def ensure_grayscale(image):
    """
    Convert an image to grayscale if it is not already.
    """
    # return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) -> Alternative short
    # check if the image has more than one channel (i.e., is not grayscale)
    if len(image.shape) > 2 and image.shape[2] > 1:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # the image is already grayscale
        return image


def draw_lines_onto_image(image, pts_prev, pts_curr):
    visualized_image = image.copy()

    # Define line thickness
    line_thickness = 1
    outline_thickness = 2

    for pt_prev, pt_curr in zip(pts_prev, pts_curr):
        start_point = (int(pt_prev[0][0]), int(pt_prev[0][1]))  # Previous image point
        end_point = (int(pt_curr[0][0]), int(pt_curr[0][1]))  # Current image point

        cv2.line(visualized_image, start_point, end_point, (0, 0, 0), outline_thickness)

        cv2.line(
            visualized_image, start_point, end_point, (255, 255, 255), line_thickness
        )

    return visualized_image


def timer(f: callable) -> callable:
    """
    Wraps a function in order to capture and print the
    execution time.

    Example
        @timer
        def f(x):
            print(x)

    """

    @wraps(f)
    def wrap(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        print(f"Function {f.__name__} took: {(end_time - start_time):.8f} seconds.")
        return result

    return wrap


@timer
def download_file(url: str) -> io.BytesIO:
    """
    Download a file from a given URL in a stream and returns it as a BytesIO object.

    :param url: URL of the file to download.
    :return: BytesIO object containing the downloaded file.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # check if the request was successful
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte

        file_stream = io.BytesIO()
        with tqdm(total=total_size, unit="iB", unit_scale=True) as progress_bar:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file_stream.write(data)

        file_stream.seek(0)  # reset stream pointer
        return file_stream


@timer
def unzip_file(file_stream: io.BytesIO, target_folder: str):
    """
    Unzip a BytesIO object to a specified target folder.

    :param file_stream: BytesIO object containing the zipped file.
    :param target_folder: Local directory path to extract the contents.
    """
    with zipfile.ZipFile(file_stream) as zipped_file:
        zipped_file.extractall(target_folder)


@timer
def download_and_unzip(url: str, target_folder: str):
    """
    Download and unzip a file from a given URL to a specified target folder.

    :param url: URL of the file to download.
    :param target_folder: Local directory path to extract the contents.
    """
    try:
        file_stream = download_file(url)
        unzip_file(file_stream, target_folder)
        print(f"Dataset extracted to {target_folder}")
    except requests.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def construct_homogeneous_matrix(R, t):
    return np.block([[R, t.reshape(-1, 1)], [np.zeros((1, 3)), np.ones((1, 1))]])


def deconstruct_homogeneous_matrix(H):
    return H[:3, :3], H[:3, 3]


def compare_poses(actual_pose, estimated_pose):
    """
    Compare two 4x4 homogeneous matrices representing poses and return the
    translation and rotation differences.

    Parameters:
    actual_pose (numpy.ndarray): The ground truth pose as a 4x4 matrix.
    estimated_pose (numpy.ndarray): The estimated pose as a 4x4 matrix.

    Returns:
    tuple: A tuple containing the translation difference (float) and the
           rotation difference in radians (float).
    """
    rotation_actual, translation_actual = deconstruct_homogeneous_matrix(actual_pose)
    rotation_estimated, translation_estimated = deconstruct_homogeneous_matrix(
        estimated_pose
    )
    # Compute translation difference
    translation_diff = np.linalg.norm(translation_actual - translation_estimated)

    # Compute rotation difference
    # Clamp value to the valid range for arccos due to possible numerical issues
    rotation_diff_val = (np.trace(rotation_actual.T @ rotation_estimated) - 1) / 2
    rotation_diff_val = np.clip(rotation_diff_val, -1.0, 1.0)
    rotation_diff = np.arccos(rotation_diff_val)

    return translation_diff, rotation_diff
