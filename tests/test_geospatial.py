import pytest
import numpy as np
from geospatial_utils import generate_tile_coords, cosine_bell_mask, stitch_predictions, apply_clahe

def test_generate_tile_coords():
    coords = generate_tile_coords(1000, 1000, 256, 0.5)

    # Bottom right tile should stop at 1000
    assert max(c[1] for c in coords) == 1000 # row max
    assert max(c[3] for c in coords) == 1000 # col max

def test_cosine_bell_mask():
    mask = cosine_bell_mask(256)
    assert mask.shape == (256, 256)
    assert np.isclose(mask.max(), 1.0)
    assert np.isclose(mask.min(), 0.0)

def test_stitch_predictions():
    # Simulate inference tiles on a 500x500 image with patch=256, stride=0.5
    H, W = 500, 500
    patch_size = 256
    overlap = 0.5

    image = np.random.rand(H, W, 4).astype(np.float32)

    class DummyModel:
        def predict(self, x, verbose):
            # return dummy predictions
            N = x.shape[0]
            preds = np.random.rand(N, patch_size, patch_size, 3)
            return preds / preds.sum(axis=-1, keepdims=True)

    model = DummyModel()

    stitched = stitch_predictions(model, image, patch_size, overlap, 4, 3)

    assert stitched.shape == (H, W)
    assert stitched.dtype == np.int32
    assert stitched.min() >= 0
    assert stitched.max() < 3

def test_apply_clahe():
    H, W, C = 64, 64, 4
    # Create a synthetic image with low contrast
    image_1d = np.linspace(0.4, 0.6, H * W)
    image_2d = image_1d.reshape(H, W)
    image = np.stack([image_2d] * C, axis=-1).astype(np.float32)

    enhanced = apply_clahe(image, clip_limit=2.0, tile_grid=(8, 8))

    # Shape should match
    assert enhanced.shape == image.shape

    # Data type should match
    assert enhanced.dtype == np.float32

    # Values should still be in [0, 1]
    assert enhanced.min() >= 0.0
    assert enhanced.max() <= 1.0

    # Contrast should be increased
    for b in range(C):
        orig_std = image[:, :, b].std()
        enh_std = enhanced[:, :, b].std()
        assert enh_std > orig_std
