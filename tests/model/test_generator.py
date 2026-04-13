import pytest
import numpy as np
from src.model.generator import _random_flip

def test_random_flip_horizontal():
    # Setup test arrays
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([[1, 2], [3, 4]])

    # Create a deterministic random number generator that will always flip horizontally (random() > 0.5) but not vertically (random() <= 0.5)
    class MockRNG:
        def __init__(self):
            self.calls = 0

        def random(self):
            self.calls += 1
            if self.calls == 1:
                return 0.6 # Horizontal flip
            return 0.4 # No vertical flip

    rng = MockRNG()

    flipped_image, flipped_mask = _random_flip(image.copy(), mask.copy(), rng)

    # Expected horizontal flip
    expected_image = np.fliplr(image)
    expected_mask = np.fliplr(mask)

    np.testing.assert_array_equal(flipped_image, expected_image)
    np.testing.assert_array_equal(flipped_mask, expected_mask)

def test_random_flip_vertical():
    # Setup test arrays
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([[1, 2], [3, 4]])

    # Create a deterministic random number generator that will not flip horizontally (random() <= 0.5) but will vertically (random() > 0.5)
    class MockRNG:
        def __init__(self):
            self.calls = 0

        def random(self):
            self.calls += 1
            if self.calls == 1:
                return 0.4 # No horizontal flip
            return 0.6 # Vertical flip

    rng = MockRNG()

    flipped_image, flipped_mask = _random_flip(image.copy(), mask.copy(), rng)

    # Expected vertical flip
    expected_image = np.flipud(image)
    expected_mask = np.flipud(mask)

    np.testing.assert_array_equal(flipped_image, expected_image)
    np.testing.assert_array_equal(flipped_mask, expected_mask)

def test_random_flip_both():
    # Setup test arrays
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([[1, 2], [3, 4]])

    # Create a deterministic random number generator that will flip both horizontally and vertically
    class MockRNG:
        def random(self):
            return 0.6 # Always flip

    rng = MockRNG()

    flipped_image, flipped_mask = _random_flip(image.copy(), mask.copy(), rng)

    # Expected both flips
    expected_image = np.flipud(np.fliplr(image))
    expected_mask = np.flipud(np.fliplr(mask))

    np.testing.assert_array_equal(flipped_image, expected_image)
    np.testing.assert_array_equal(flipped_mask, expected_mask)

def test_random_flip_none():
    # Setup test arrays
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([[1, 2], [3, 4]])

    # Create a deterministic random number generator that will not flip
    class MockRNG:
        def random(self):
            return 0.4 # Never flip

    rng = MockRNG()

    flipped_image, flipped_mask = _random_flip(image.copy(), mask.copy(), rng)

    # Expected no flips
    np.testing.assert_array_equal(flipped_image, image)
    np.testing.assert_array_equal(flipped_mask, mask)

def test_random_flip_invariants():
    image = np.random.rand(10, 10, 4)
    mask = np.random.randint(0, 3, size=(10, 10))
    rng = np.random.default_rng(42)
    flipped_image, flipped_mask = _random_flip(image.copy(), mask.copy(), rng)

    assert flipped_image.shape == image.shape
    assert flipped_mask.shape == mask.shape
    assert np.isclose(np.mean(flipped_image), np.mean(image))
    assert np.isclose(np.mean(flipped_mask), np.mean(mask))
    assert np.isclose(np.sum(flipped_image), np.sum(image))
    assert np.isclose(np.sum(flipped_mask), np.sum(mask))
