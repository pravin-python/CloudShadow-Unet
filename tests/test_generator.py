import pytest
import numpy as np
import tensorflow as tf
from src.model.generator import _random_rotate90

class MockRNG:
    def __init__(self, value):
        self.value = value
    def integers(self, low, high):
        return self.value

def test_random_rotate90_0():
    rng = MockRNG(0)
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([[1, 2], [3, 4]])
    out_image, out_mask = _random_rotate90(image, mask, rng)
    np.testing.assert_array_equal(out_image, image)
    np.testing.assert_array_equal(out_mask, mask)

def test_random_rotate90_1():
    rng = MockRNG(1)
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([[1, 2], [3, 4]])
    # A 90 degree CCW rotation should move the top-right to top-left, bottom-right to top-right, etc.
    expected_image = np.rot90(image, k=1, axes=(0, 1))
    expected_mask = np.rot90(mask, k=1)

    out_image, out_mask = _random_rotate90(image, mask, rng)
    np.testing.assert_array_equal(out_image, expected_image)
    np.testing.assert_array_equal(out_mask, expected_mask)

def test_random_rotate90_2():
    rng = MockRNG(2)
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([[1, 2], [3, 4]])

    expected_image = np.rot90(image, k=2, axes=(0, 1))
    expected_mask = np.rot90(mask, k=2)

    out_image, out_mask = _random_rotate90(image, mask, rng)
    np.testing.assert_array_equal(out_image, expected_image)
    np.testing.assert_array_equal(out_mask, expected_mask)

def test_random_rotate90_3():
    rng = MockRNG(3)
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([[1, 2], [3, 4]])

    expected_image = np.rot90(image, k=3, axes=(0, 1))
    expected_mask = np.rot90(mask, k=3)

    out_image, out_mask = _random_rotate90(image, mask, rng)
    np.testing.assert_array_equal(out_image, expected_image)
    np.testing.assert_array_equal(out_mask, expected_mask)

def test_random_rotate90_preserves_dtype():
    rng = MockRNG(1)
    image = np.zeros((10, 10, 4), dtype=np.float32)
    mask = np.zeros((10, 10), dtype=np.uint8)

    out_image, out_mask = _random_rotate90(image, mask, rng)
    assert out_image.dtype == image.dtype
    assert out_mask.dtype == mask.dtype
