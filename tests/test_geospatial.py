import pytest
import numpy as np
from geospatial_utils import generate_tile_coords, cosine_bell_mask, stitch_predictions

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


def test_compute_area_stats():
    from geospatial_utils import compute_area_stats

    # 0 = Background, 1 = Cloud, 2 = Shadow
    # 10x10 array = 100 pixels
    # Let's say: 50 background, 30 cloud, 20 shadow
    class_map = np.zeros((10, 10), dtype=np.int32)
    class_map[0:3, :] = 1 # 30 cloud pixels
    class_map[3:5, :] = 2 # 20 shadow pixels

    # Using default pixel_area_m2 = 100.0 (100 pixels = 10000 m^2 = 0.01 km^2)
    stats = compute_area_stats(class_map)

    # Background: 50 pixels, 50 * 100 = 5000 m^2 = 0.0050 km^2, 50%
    assert stats['Background']['px_count'] == 50.0
    assert stats['Background']['area_km2'] == 0.0050
    assert stats['Background']['percentage'] == 50.0

    # Cloud: 30 pixels, 30 * 100 = 3000 m^2 = 0.0030 km^2, 30%
    assert stats['Cloud']['px_count'] == 30.0
    assert stats['Cloud']['area_km2'] == 0.0030
    assert stats['Cloud']['percentage'] == 30.0

    # Shadow: 20 pixels, 20 * 100 = 2000 m^2 = 0.0020 km^2, 20%
    assert stats['Shadow']['px_count'] == 20.0
    assert stats['Shadow']['area_km2'] == 0.0020
    assert stats['Shadow']['percentage'] == 20.0

def test_compute_area_stats_custom_area():
    from geospatial_utils import compute_area_stats

    class_map = np.zeros((10, 10), dtype=np.int32)

    # 10x10 array = 100 pixels. Each pixel = 900 m^2. Total = 90000 m^2 = 0.09 km^2
    stats = compute_area_stats(class_map, pixel_area_m2=900.0)

    assert stats['Background']['px_count'] == 100.0
    assert stats['Background']['area_km2'] == 0.0900
    assert stats['Background']['percentage'] == 100.0

    assert stats['Cloud']['px_count'] == 0.0
    assert stats['Cloud']['area_km2'] == 0.0
    assert stats['Cloud']['percentage'] == 0.0

def test_compute_area_stats_missing_class():
    from geospatial_utils import compute_area_stats

    # All background and cloud, NO shadow
    class_map = np.zeros((10, 10), dtype=np.int32)
    class_map[0:5, :] = 1 # 50 cloud pixels

    stats = compute_area_stats(class_map)

    assert stats['Background']['px_count'] == 50.0
    assert stats['Cloud']['px_count'] == 50.0
    assert stats['Shadow']['px_count'] == 0.0

    assert stats['Shadow']['area_km2'] == 0.0
    assert stats['Shadow']['percentage'] == 0.0

def test_compute_area_stats_zero_total_area():
    from geospatial_utils import compute_area_stats

    class_map = np.empty((0, 0), dtype=np.int32)

    stats = compute_area_stats(class_map)

    assert stats['Background']['px_count'] == 0.0
    assert stats['Background']['area_km2'] == 0.0
    assert stats['Background']['percentage'] == 0.0 # Handled by max(total_px, 1)
