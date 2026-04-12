"""
scripts/create_synthetic_demo.py
=================================
Generates a fully synthetic 4-band GeoTIFF scene + ground-truth mask
so you can run the entire pipeline immediately — no internet or dataset
download required.

Generates:
  data/raw/demo_scene.tif        ← uint16 (512, 512, 4) GeoTIFF  (RGBNIR)
  data/raw/demo_scene_mask.tif   ← uint8  (512, 512)    GeoTIFF  (0/1/2)

The synthetic scene contains:
  • A gradient terrain background (class 0)
  • Two elliptical cloud blobs (class 1)
  • Shadow regions offset from each cloud (class 2)

After running this script, you can immediately run:
  python geospatial_utils.py --image data/raw/demo_scene.tif ...
  python train.py --mode train ...
  streamlit run app.py

Usage:
  python scripts/create_synthetic_demo.py
  python scripts/create_synthetic_demo.py --size 1024 --n_clouds 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"


def _draw_ellipse_mask(
    canvas: np.ndarray,
    cx: int, cy: int,
    rx: int, ry: int,
    value: int,
) -> None:
    """Draw a filled ellipse on canvas with the given label value (in-place)."""
    h, w = canvas.shape
    y, x = np.ogrid[:h, :w]
    mask = ((x - cx) ** 2 / rx ** 2 + (y - cy) ** 2 / ry ** 2) <= 1.0
    canvas[mask] = value


def create_synthetic_scene(
    size: int = 512,
    n_clouds: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic (H, W, 4) uint16 scene and (H, W) uint8 mask.

    Args:
        size:     Spatial dimension (size × size pixels).
        n_clouds: Number of cloud blobs to place.
        seed:     NumPy random seed for reproducibility.

    Returns:
        (image_uint16, mask_uint8) tuple.
        image_uint16: float32 reflectance values scaled to [0, 10000] uint16.
        mask_uint8:   Labels {0=Background, 1=Cloud, 2=Shadow}.
    """
    rng = np.random.default_rng(seed)
    H = W = size

    # ── Background reflectance (vegetated terrain gradient) ───────────────────
    # Each band has a smooth spatial gradient + small-scale noise
    # to simulate realistic heterogeneous terrain
    band_base = {
        "red":   (0.08, 0.18),   # Low red → vegetation
        "green": (0.09, 0.19),
        "blue":  (0.05, 0.14),
        "nir":   (0.25, 0.50),   # High NIR → vegetation (red edge effect)
    }

    bands: dict[str, np.ndarray] = {}
    for band_name, (lo, hi) in band_base.items():
        # Gradient across the image
        grad_y = np.linspace(lo, hi, H)[:, np.newaxis] * np.ones((H, W))
        grad_x = np.linspace(lo, hi, W)[np.newaxis, :] * np.ones((H, W))
        gradient = (grad_y + grad_x) / 2.0

        # Small spatial noise
        noise = rng.normal(0, 0.01, size=(H, W))
        bands[band_name] = np.clip(gradient + noise, 0.0, 1.0).astype(np.float32)

    # ── Mask: start all background ─────────────────────────────────────────────
    mask = np.zeros((H, W), dtype=np.uint8)

    # ── Place clouds + shadows ────────────────────────────────────────────────
    for _ in range(n_clouds):
        cx  = int(rng.integers(size // 5, 4 * size // 5))
        cy  = int(rng.integers(size // 5, 4 * size // 5))
        rx  = int(rng.integers(size // 12, size // 5))
        ry  = int(rng.integers(size // 16, size // 6))

        # Shadow: offset from cloud centre (shadow cast direction ~SE)
        shadow_offset_x = int(rx * 0.6)
        shadow_offset_y = int(ry * 0.6)
        _draw_ellipse_mask(mask, cx + shadow_offset_x, cy + shadow_offset_y,
                           int(rx * 0.8), int(ry * 0.8), value=2)

        # Cloud blob on top of (erases) shadow pixels where they overlap
        _draw_ellipse_mask(mask, cx, cy, rx, ry, value=1)

        # Modify pixel values inside cloud — high reflectance in all bands
        cloud_mask = (mask == 1)
        for b in bands.values():
            base_val = float(rng.uniform(0.75, 0.95))
            cloud_noise = rng.normal(0, 0.02, size=(H, W)).astype(np.float32)
            b[cloud_mask] = np.clip(base_val + cloud_noise[cloud_mask], 0.8, 1.0)

        # Shadow pixels — darker in all bands, especially NIR
        shadow_mask = (mask == 2)
        for bname, b in bands.items():
            dark_factor = 0.25 if bname == "nir" else 0.45
            b[shadow_mask] = b[shadow_mask] * dark_factor

    # ── Convert reflectance [0,1] → uint16 [0,10000] ─────────────────────────
    stack = np.stack(
        [bands["red"], bands["green"], bands["blue"], bands["nir"]],
        axis=-1,
    )
    image_uint16 = (np.clip(stack, 0.0, 1.0) * 10_000.0).astype(np.uint16)

    # Log class distribution
    total = mask.size
    for cls_id, name in {0: "Background", 1: "Cloud", 2: "Shadow"}.items():
        cnt = int(np.sum(mask == cls_id))
        logger.info("  %-12s: %6d px  (%.1f%%)", name, cnt, 100.0 * cnt / total)

    return image_uint16, mask


def save_geotiff_scene(
    image_uint16: np.ndarray,
    mask_uint8:   np.ndarray,
    out_image:    Path,
    out_mask:     Path,
) -> None:
    """Save the synthetic scene pair as valid GeoTIFFs with a WGS84 CRS.

    The CRS is EPSG:32632 (UTM Zone 32N) — a realistic Sentinel-2 CRS
    for Central Europe.  The affine transform places the scene at a
    plausible UTM coordinate with 10 m GSD (matching Sentinel-2 Band 4/8).

    Args:
        image_uint16: (H, W, 4) uint16 reflectance array.
        mask_uint8:   (H, W)    uint8  class label array.
        out_image:    Destination path for the image GeoTIFF.
        out_mask:     Destination path for the mask GeoTIFF.
    """
    H, W = image_uint16.shape[:2]

    # Realistic UTM Zone 32N coordinates for demo scene
    transform = from_origin(
        west=399_960.0,    # Easting (m) — typical Sentinel-2 tile origin
        north=5_300_040.0, # Northing (m)
        xsize=10.0,        # 10 m pixel width
        ysize=10.0,        # 10 m pixel height
    )
    crs = rasterio.crs.CRS.from_epsg(32632)   # WGS 84 / UTM Zone 32N

    out_image.parent.mkdir(parents=True, exist_ok=True)

    # ── Image GeoTIFF ─────────────────────────────────────────────────────────
    img_profile = {
        "driver":   "GTiff",
        "dtype":    "uint16",
        "count":    4,
        "height":   H,
        "width":    W,
        "crs":      crs,
        "transform": transform,
        "compress": "lzw",
    }
    with rasterio.open(out_image, "w", **img_profile) as dst:
        # rasterio expects (bands, H, W)
        dst.write(np.transpose(image_uint16, (2, 0, 1)))
        dst.update_tags(
            BAND_1="Red (0.665 μm)",
            BAND_2="Green (0.560 μm)",
            BAND_3="Blue (0.490 μm)",
            BAND_4="NIR (0.842 μm)",
            DESCRIPTION="Synthetic demo scene — CloudShadow-UNet",
        )
    logger.info("Image GeoTIFF → %s", out_image)

    # ── Mask GeoTIFF ──────────────────────────────────────────────────────────
    mask_profile = {
        "driver":   "GTiff",
        "dtype":    "uint8",
        "count":    1,
        "height":   H,
        "width":    W,
        "crs":      crs,
        "transform": transform,
        "compress": "lzw",
        "nodata":   255,
    }
    with rasterio.open(out_mask, "w", **mask_profile) as dst:
        dst.write(mask_uint8[np.newaxis, :, :])
        dst.write_colormap(1, {
            0:   (128, 128, 128, 255),
            1:   (255, 255, 255, 255),
            2:   (30,  60,  120, 255),
            255: (0, 0, 0, 0),
        })
        dst.update_tags(
            BAND_1="Class labels: 0=Background, 1=Cloud, 2=Shadow",
            DESCRIPTION="Synthetic demo mask — CloudShadow-UNet",
        )
    logger.info("Mask GeoTIFF → %s", out_mask)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic demo GeoTIFF scene + mask",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size",      type=int, default=512,  help="Scene side length in pixels")
    parser.add_argument("--n_clouds",  type=int, default=3,    help="Number of cloud blobs")
    parser.add_argument("--n_scenes",  type=int, default=5,    help="Number of distinct scenes to generate")
    parser.add_argument("--seed",      type=int, default=42,   help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_scenes):
        scene_seed = args.seed + i
        logger.info("Generating scene %d/%d (seed=%d) …", i + 1, args.n_scenes, scene_seed)

        image_u16, mask_u8 = create_synthetic_scene(
            size=args.size,
            n_clouds=args.n_clouds,
            seed=scene_seed,
        )
        save_geotiff_scene(
            image_u16, mask_u8,
            out_image = RAW_DIR / f"demo_scene_{i+1:03d}.tif",
            out_mask  = RAW_DIR / f"demo_scene_{i+1:03d}_mask.tif",
        )

    logger.info(
        "\n✅ Done! Generated %d scene(s) in data/raw/\n"
        "   Next steps:\n"
        "   1. python geospatial_utils.py  (preprocess patches)\n"
        "   2. python train.py --mode train\n"
        "   3. streamlit run app.py",
        args.n_scenes,
    )
