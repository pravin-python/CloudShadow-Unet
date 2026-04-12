import pytest
import numpy as np
from geospatial_utils import generate_tile_coords, cosine_bell_mask, stitch_predictions, class_map_to_rgb, CLASS_COLORS_RGB

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

def test_class_map_to_rgb():
    # Test typical case
    class_map = np.array([
        [0, 1],
        [2, 0]
    ], dtype=np.int32)

    rgb = class_map_to_rgb(class_map)

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8

    expected_0 = CLASS_COLORS_RGB[0]
    expected_1 = CLASS_COLORS_RGB[1]
    expected_2 = CLASS_COLORS_RGB[2]

    assert np.array_equal(rgb[0, 0], expected_0)
    assert np.array_equal(rgb[0, 1], expected_1)
    assert np.array_equal(rgb[1, 0], expected_2)
    assert np.array_equal(rgb[1, 1], expected_0)

def test_class_map_to_rgb_edge_cases():
    # Test empty array
    empty_map = np.empty((0, 0), dtype=np.int32)
    rgb_empty = class_map_to_rgb(empty_map)
    assert rgb_empty.shape == (0, 0, 3)
    assert rgb_empty.dtype == np.uint8

    # Test array with unexpected values
    unexpected_map = np.array([
        [3, -1],
        [0, 99]
    ], dtype=np.int32)

    rgb_unexpected = class_map_to_rgb(unexpected_map)
    assert rgb_unexpected.shape == (2, 2, 3)
    assert rgb_unexpected.dtype == np.uint8

    # 0 should map correctly
    assert np.array_equal(rgb_unexpected[1, 0], CLASS_COLORS_RGB[0])

    # 3, -1, 99 should remain black (0, 0, 0)
    assert np.array_equal(rgb_unexpected[0, 0], [0, 0, 0])
    assert np.array_equal(rgb_unexpected[0, 1], [0, 0, 0])
    assert np.array_equal(rgb_unexpected[1, 1], [0, 0, 0])
