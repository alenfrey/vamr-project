from hypothesis import given, strategies as st
from src.utils import construct_homogeneous_matrix
import numpy as np

# define strategies for generating random rotation matrices and translation vectors
rotation_matrix_strategy = st.lists(
    st.lists(st.floats(), min_size=3, max_size=3), min_size=3, max_size=3
)
translation_vector_strategy = st.lists(st.floats(), min_size=3, max_size=3)


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
