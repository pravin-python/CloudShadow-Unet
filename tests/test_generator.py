import pytest
import numpy as np
import tensorflow as tf
from src.model.generator import _random_brightness

def test_random_brightness_changes_image():
    """Test that _random_brightness modifies the image properly."""
    rng = np.random.default_rng(42)
    # Create an image filled with 0.5 (mid-gray)
    image = np.ones((256, 256, 4), dtype=np.float32) * 0.5

    # Store original max delta
    max_delta = 0.2

    # Apply brightness augmentation
    aug_image = _random_brightness(image, rng, max_delta=max_delta)

    # Check that image changed
    assert not np.array_equal(image, aug_image)

    # Check that delta is uniform across the image
    # (since delta is a single scalar added to the entire image)
    delta_image = aug_image - image
    # Note: Because of clipping, delta might not be exactly equal to the scalar if clipping happened,
    # but since image is 0.5 and delta <= 0.2, clipping shouldn't happen here.
    assert np.allclose(delta_image, delta_image[0,0,0], atol=1e-5)

    # Check that values are within bounds
    assert np.all(aug_image >= 0.0)
    assert np.all(aug_image <= 1.0)

def test_random_brightness_clipping():
    """Test that _random_brightness correctly clips values to [0.0, 1.0]."""
    # Fix the seed such that rng.uniform(-max_delta, max_delta) is positive
    # We will test an image filled with 0.95 and max_delta 0.1 to trigger clipping at 1.0
    rng = np.random.default_rng(1) # Seed 1 gives a positive first uniform sample

    image = np.ones((256, 256, 4), dtype=np.float32) * 0.95
    aug_image = _random_brightness(image, rng, max_delta=0.1)

    assert np.all(aug_image <= 1.0)

    # Test clipping at 0.0
    rng = np.random.default_rng(42) # Seed 42 gives a negative first uniform sample
    image = np.ones((256, 256, 4), dtype=np.float32) * 0.05
    aug_image = _random_brightness(image, rng, max_delta=0.1)

    assert np.all(aug_image >= 0.0)
