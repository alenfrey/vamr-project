import pytest
import numpy as np
import sys

from hypothesis import given, strategies as st
from src.utils import *


valid_rotation_matrices = [
    np.eye(3),
    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    np.array(
        [
            [1, 0, 0],
            [0, np.sqrt(2) / 2, -np.sqrt(2) / 2],
            [0, np.sqrt(2) / 2, np.sqrt(2) / 2],
        ]
    ),
    np.array(
        [
            [np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
            [0, 1, 0],
            [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
        ]
    ),
    np.array(
        [
            [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
            [np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
            [0, 0, 1],
        ]
    ),
]


rotation_matrix_strategy = st.sampled_from(valid_rotation_matrices).filter(
    is_valid_rotation_matrix
)

# Updated strategy for translation vectors
translation_vector_strategy = st.lists(
    st.floats(min_value=-100, max_value=100), min_size=3, max_size=3
)

# Updated strategy for points_3d
points_3d_strategy = st.lists(
    st.lists(st.floats(min_value=-100, max_value=100), min_size=3, max_size=3),
    min_size=1,
)


def test_valid_rotation_matrices():
    # Positive test with identity matrix
    R_identity = np.eye(3)
    assert is_valid_rotation_matrix(R_identity)

    # Positive test with a known valid rotation matrix
    R_90_z = np.array(
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    )  # 90-degree rotation about z-axis
    assert is_valid_rotation_matrix(R_90_z)

    # Add more positive tests with other known valid rotation matrices


def test_invalid_rotation_matrices():
    # Negative test with a non-square matrix
    R_non_square = np.array([[1, 0], [0, 1], [0, 0]])
    assert not is_valid_rotation_matrix(R_non_square)

    # Negative test with a square but not orthogonal matrix
    R_not_orthogonal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert not is_valid_rotation_matrix(R_not_orthogonal)

    # Negative test with orthogonal but wrong determinant
    R_wrong_determinant = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
    )  # determinant should be -1
    assert not is_valid_rotation_matrix(R_wrong_determinant)

    # Add more negative tests with other invalid matrices


def test_edge_case_rotation_matrices():
    # Edge case with a matrix very close to a valid rotation matrix
    R_almost_valid = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1 - 1e-5]]
    )  # Almost an identity matrix
    assert not is_valid_rotation_matrix(R_almost_valid)


@given(rotation_matrix_strategy, translation_vector_strategy)
def test_construct_homogeneous_matrix(R, t):
    R = np.array(R)
    t = np.array(t)
    T = construct_homogeneous_matrix(R, t)

    # check that T is a 4x4 matrix
    assert T.shape == (4, 4)

    # check that the top-left 3x3 sub-matrix is equal to R
    np.testing.assert_array_almost_equal(T[:3, :3], R)

    # check that the first three elements of the last column are equal to t
    np.testing.assert_array_almost_equal(T[:3, 3], t.ravel())

    # check that the last row is [0, 0, 0, 1]
    np.testing.assert_array_almost_equal(T[3, :], np.array([0, 0, 0, 1]))


@given(rotation_matrix_strategy, translation_vector_strategy)
def test_deconstruct_homogeneous_matrix(R, t):
    R = np.array(R)
    t = np.array(t)
    T = construct_homogeneous_matrix(R, t)

    R_deconstructed, t_deconstructed = deconstruct_homogeneous_matrix(T)
    np.testing.assert_array_almost_equal(R, R_deconstructed)
    np.testing.assert_array_almost_equal(t.ravel(), t_deconstructed)


@given(rotation_matrix_strategy, translation_vector_strategy)
def test_compare_poses_identical(R, t):
    R = np.array(R)
    t = np.array(t)
    T = construct_homogeneous_matrix(R, t)

    translation_diff, rotation_diff = compare_poses(T, T)
    assert translation_diff == 0
    assert rotation_diff == 0


def test_compare_poses_known_difference():
    R1 = np.eye(3)
    t1 = np.array([0, 0, 0])
    T1 = construct_homogeneous_matrix(R1, t1)

    R2 = np.eye(3)
    t2 = np.array([1, 0, 0])
    T2 = construct_homogeneous_matrix(R2, t2)

    translation_diff, rotation_diff = compare_poses(T1, T2)
    assert translation_diff == 1
    assert rotation_diff == 0


@given(points_3d_strategy, rotation_matrix_strategy, translation_vector_strategy)
def test_to_world_coordinates_with_pose(points_3d, R, t):
    points_3d = np.array(points_3d).reshape(-1, 3).T
    R = np.array(R)
    t = np.array(t).reshape(3, 1)
    pose = np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])

    transformed_points = to_world_coordinates(points_3d, pose=pose)
    expected_transformed_points = pose[:3, :3] @ points_3d + pose[:3, 3:4]
    np.testing.assert_array_almost_equal(
        transformed_points, expected_transformed_points
    )


@given(points_3d_strategy, rotation_matrix_strategy, translation_vector_strategy)
def test_to_world_coordinates_with_R_and_t(points_3d, R, t):
    points_3d = np.array(points_3d).reshape(-1, 3).T
    R = np.array(R)
    t = np.array(t).reshape(3, 1)

    transformed_points = to_world_coordinates(points_3d, R=R, t=t)
    expected_transformed_points = R @ points_3d + t
    np.testing.assert_array_almost_equal(
        transformed_points, expected_transformed_points
    )


def test_to_world_coordinates_invalid_arguments():
    points_3d = np.random.rand(3, 3)
    R = np.eye(3)
    t = np.zeros((3, 1))
    pose = np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])

    # Test with neither pose nor R and t
    with pytest.raises(ValueError):
        to_world_coordinates(points_3d)

    # Test with both pose and R
    with pytest.raises(ValueError):
        to_world_coordinates(points_3d, pose=pose, R=R)
