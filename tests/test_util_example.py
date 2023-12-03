from hypothesis import given, strategies as st
from util import add_numbers


@given(st.integers(), st.integers())
def test_add_numbers(x, y):
    assert add_numbers(x, y) == x + y
