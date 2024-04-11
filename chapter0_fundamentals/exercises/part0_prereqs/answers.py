import sys
import numpy as np
import einops
from pathlib import Path

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part0_prereqs', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part0_prereqs.utils import display_array_as_img
import part0_prereqs.tests as tests

MAIN = __name__ == "__main__"

arr = np.load(section_dir / "numbers.npy")

display_array_as_img(arr[0])

# Exercise 1
display_array_as_img(einops.rearrange(arr, 'b c h w -> c h (b w)'))

# Exercise 2
display_array_as_img(einops.repeat(arr[0], 'c h w -> c (2 h) w'))

# Exercise 3
display_array_as_img(einops.repeat(einops.rearrange(arr[0:2], 'b c h w -> c (b h) w'), 'c h w -> c h (2 w)'))

# Exercise 4
display_array_as_img(einops.repeat(arr[0], 'c h w -> c (h 2) w'))

# Exercise 5
display_array_as_img(einops.rearrange(arr[0], 'c h w -> h (c w)'))

# Exercise 6
display_array_as_img(einops.rearrange(arr, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=2))

# Exercise 7
display_array_as_img(einops.reduce(arr, 'b c h w -> h (b w)', 'max'))

# Exercise 8
display_array_as_img(einops.reduce(arr, 'b c h w -> h w', 'min'))

# Exercise 9
display_array_as_img(einops.rearrange(arr[1], 'c h w -> c w h'))

# Exercise 10
display_array_as_img(einops.reduce(arr, '(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)', 'max', b1=2, h2=2, w2=2))

import torch as t

def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Passed!")

def assert_all_close(actual: t.Tensor, expected: t.Tensor, rtol=1e-05, atol=0.0001) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert t.allclose(actual, expected, rtol=rtol, atol=atol)
    print("Passed!")

def rearrange_1() -> t.Tensor:
    '''Return the following tensor using only torch.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    '''
    return einops.rearrange(t.arange(3, 9), "(h w) -> h w", h=3)

expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)

def rearrange_2() -> t.Tensor:
    '''Return the following tensor using only torch.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    '''
    return einops.rearrange(t.arange(1, 7), "(h w) -> h w", h=2)

assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))

def rearrange_3() -> t.Tensor:
    '''Return the following tensor using only torch.arange and einops.rearrange:

    [[[1], [2], [3], [4], [5], [6]]]
    '''
    return einops.rearrange(t.arange(1, 7), "w -> 1 w 1")

assert_all_equal(rearrange_3(), t.tensor([[[1], [2], [3], [4], [5], [6]]]))

def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    return einops.einsum(mat, "i i ->")

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    return einops.einsum(mat, vec, "i j, j -> i")

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return einops.einsum(mat1, mat2, "i j, j k -> i k")

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return einops.einsum(vec1, vec2, "i, i ->")

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return einops.einsum(vec1, vec2, "i, j -> i j")


tests.test_einsum_trace(einsum_trace)
tests.test_einsum_mv(einsum_mv)
tests.test_einsum_mm(einsum_mm)
tests.test_einsum_inner(einsum_inner)
tests.test_einsum_outer(einsum_outer)