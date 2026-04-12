"""
Module 1 — Geospatial Data Preprocessing
=========================================
Reads 4-band (R, G, B, NIR) 16-bit GeoTIFF imagery via rasterio,
normalises pixel intensities to [0, 1], applies per-band CLAHE contrast
enhancement, and tiles large scenes into overlapping 256×256 patches
saved as NumPy arrays alongside their corresponding ground-truth masks.

Usage (CLI):
    python src/preprocessing/preprocess.py \
        --image  data/raw/scene.tif \
        --mask   data/raw/scene_mask.tif \
        --out_img  data/patches \
        --out_mask data/masks \
        --patch_size 256 \
        --overlap 0.25
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── constants ────────────────────────────────────────────────────────────────
# 38-Cloud / Sentinel-2 Level-2A reflectance is stored as UInt16
# Typical valid range: 0–10000 (scale factor 10000 = 100% reflectance)
REFLECTANCE_SCALE: float = 10_000.0

# CLAHE parameters — tuned for thin cirrus detection
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def read_multiband_geotiff(path: Path) -> tuple[np.ndarray, dict]:
    """Read a 4-band GeoTIFF and return a (H, W, 4) float32 array + profile.

    Bands are expected in order: Red, Green, Blue, NIR.
    Raw UInt16 reflectance values are divided by REFLECTANCE_SCALE and
    clipped to [0, 1] to handle occasional sensor saturation artefacts.

    Args:
        path: Absolute path to the source GeoTIFF.

    Returns:
        image:   float32 array of shape (H, W, 4), range [0, 1].
        profile: rasterio dataset profile (CRS, transform, dtype …).

    Raises:
        ValueError: If the file has fewer than 4 bands.
    """
    with rasterio.open(path) as src:
        if src.count < 4:
            raise ValueError(
                f"{path} has only {src.count} band(s); 4 (R,G,B,NIR) required."
            )
        # Read bands 1–4; rasterio returns (bands, H, W)
        raw: np.ndarray = src.read(
            [1, 2, 3, 4],
            out_dtype=np.float32,
            resampling=Resampling.bilinear,
        )
        profile = src.profile.copy()

    # (bands, H, W) → (H, W, bands)
    image = np.transpose(raw, (1, 2, 0))
    image = np.clip(image / REFLECTANCE_SCALE, 0.0, 1.0)

    logger.info(
        "Loaded %s — shape=%s  min=%.4f  max=%.4f",
        path.name, image.shape, image.min(), image.max(),
    )
    return image, profile


def read_mask_geotiff(path: Path) -> np.ndarray:
    """Read a single-band integer mask GeoTIFF.

    Expected label encoding:
        0 → Background
        1 → Cloud
        2 → Cloud Shadow

    Args:
        path: Absolute path to the mask GeoTIFF.

    Returns:
        mask: uint8 array of shape (H, W).
    """
    with rasterio.open(path) as src:
        mask: np.ndarray = src.read(1).astype(np.uint8)
    logger.info("Loaded mask %s — shape=%s  unique=%s", path.name, mask.shape, np.unique(mask))
    return mask


# ─── normalisation & enhancement ──────────────────────────────────────────────

def apply_clahe_per_band(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE independently to each of the 4 bands.

    Input is float32 [0, 1].  CLAHE requires 8-bit or 16-bit integer input,
    so we temporarily promote to UInt16, apply CLAHE, then rescale back.

    Why CLAHE?  Thin cirrus clouds often occupy a narrow slice of the
    reflectance histogram; CLAHE redistributes contrast locally so the model
    sees sharper cloud edges, especially in the NIR band.

    Args:
        image: float32 (H, W, 4) array in [0, 1].

    Returns:
        enhanced: float32 (H, W, 4) array in [0, 1].
    """
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )
    enhanced = np.empty_like(image)
    for b in range(image.shape[2]):
        # Scale to uint16 range for CLAHE
        band_u16 = (image[:, :, b] * 65535).astype(np.uint16)
        band_eq = clahe.apply(band_u16)
        enhanced[:, :, b] = band_eq.astype(np.float32) / 65535.0
    return enhanced


# ─── spatial tiling ───────────────────────────────────────────────────────────

def generate_patch_coords(
    height: int,
    width: int,
    patch_size: int = 256,
    overlap: float = 0.25,
) -> list[tuple[int, int, int, int]]:
    """Generate (row_start, row_end, col_start, col_end) patch coordinates.

    The final row / column patch is adjusted (shifted left/up) to always
    produce full-sized patches, avoiding partial tiles at boundaries.

    Args:
        height:     Image height in pixels.
        width:      Image width in pixels.
        patch_size: Side length of each square patch (default 256).
        overlap:    Fractional overlap between adjacent patches (default 0.25).

    Returns:
        List of (r0, r1, c0, c1) tuples defining each patch window.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1); got {overlap}")

    stride = int(patch_size * (1.0 - overlap))
    coords: list[tuple[int, int, int, int]] = []

    r = 0
    while r < height:
        r0 = min(r, height - patch_size)
        r1 = r0 + patch_size
        c = 0
        while c < width:
            c0 = min(c, width - patch_size)
            c1 = c0 + patch_size
            coords.append((r0, r1, c0, c1))
            if c + stride >= width:
                break
            c += stride
        if r + stride >= height:
            break
        r += stride

    return coords


def tile_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray | None,
    patch_size: int = 256,
    overlap: float = 0.25,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    """Slice image (and optionally mask) into overlapping square patches.

    Args:
        image:      float32 (H, W, 4) normalised image.
        mask:       uint8  (H, W)     ground-truth mask (optional).
        patch_size: Patch side length.
        overlap:    Fractional stride overlap.

    Returns:
        img_patches:  List of float32 (patch_size, patch_size, 4) arrays.
        mask_patches: List of uint8  (patch_size, patch_size)    arrays,
                      or None if mask is None.
    """
    h, w = image.shape[:2]
    coords = generate_patch_coords(h, w, patch_size, overlap)

    img_patches: list[np.ndarray] = []
    mask_patches: list[np.ndarray] | None = [] if mask is not None else None

    for r0, r1, c0, c1 in coords:
        img_patches.append(image[r0:r1, c0:c1, :])
        if mask_patches is not None and mask is not None:
            mask_patches.append(mask[r0:r1, c0:c1])

    logger.info(
        "Tiled into %d patches of %d×%d (overlap=%.0f%%)",
        len(img_patches), patch_size, patch_size, overlap * 100,
    )
    return img_patches, mask_patches


# ─── save helpers ─────────────────────────────────────────────────────────────

def save_patches(
    img_patches: list[np.ndarray],
    mask_patches: list[np.ndarray] | None,
    out_img_dir: Path,
    out_mask_dir: Path,
    scene_stem: str,
) -> None:
    """Persist patch arrays as compressed .npy files.

    Args:
        img_patches:  Image patch list.
        mask_patches: Mask patch list (may be None for inference-only).
        out_img_dir:  Destination directory for image patches.
        out_mask_dir: Destination directory for mask patches.
        scene_stem:   Base name prefix (typically the GeoTIFF stem).
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    for idx, patch in enumerate(img_patches):
        np.save(out_img_dir / f"{scene_stem}_patch_{idx:05d}.npy", patch)

    if mask_patches is not None:
        for idx, mpatch in enumerate(mask_patches):
            np.save(out_mask_dir / f"{scene_stem}_patch_{idx:05d}.npy", mpatch)

    logger.info(
        "Saved %d image patches → %s", len(img_patches), out_img_dir
    )


# ─── main pipeline ────────────────────────────────────────────────────────────

def preprocess_scene(
    image_path: Path,
    mask_path: Path | None,
    out_img_dir: Path,
    out_mask_dir: Path,
    patch_size: int = 256,
    overlap: float = 0.25,
    apply_clahe: bool = True,
) -> None:
    """Full preprocessing pipeline for a single scene.

    Steps:
        1. Read 4-band GeoTIFF → float32 [0, 1]
        2. (Optional) Apply per-band CLAHE enhancement
        3. Read ground-truth mask (if provided)
        4. Tile image + mask into overlapping patches
        5. Save patches as .npy files

    Args:
        image_path:   Path to source 4-band GeoTIFF.
        mask_path:    Path to mask GeoTIFF (None for inference-only scenes).
        out_img_dir:  Output directory for image patches.
        out_mask_dir: Output directory for mask patches.
        patch_size:   Patch side length in pixels.
        overlap:      Fractional overlap between adjacent patches.
        apply_clahe:  Whether to run CLAHE contrast enhancement.
    """
    image, _ = read_multiband_geotiff(image_path)

    if apply_clahe:
        logger.info("Applying CLAHE contrast enhancement …")
        image = apply_clahe_per_band(image)

    mask = read_mask_geotiff(mask_path) if mask_path is not None else None

    img_patches, mask_patches = tile_image_and_mask(
        image, mask, patch_size=patch_size, overlap=overlap
    )

    save_patches(
        img_patches,
        mask_patches,
        out_img_dir,
        out_mask_dir,
        scene_stem=image_path.stem,
    )


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a 4-band GeoTIFF scene into patch arrays."
    )
    parser.add_argument("--image", required=True, type=Path, help="Source 4-band GeoTIFF")
    parser.add_argument("--mask", type=Path, default=None, help="Ground-truth mask GeoTIFF")
    parser.add_argument("--out_img", type=Path, default=Path("data/patches"))
    parser.add_argument("--out_mask", type=Path, default=Path("data/masks"))
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--no_clahe", action="store_true", help="Skip CLAHE")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess_scene(
        image_path=args.image,
        mask_path=args.mask,
        out_img_dir=args.out_img,
        out_mask_dir=args.out_mask,
        patch_size=args.patch_size,
        overlap=args.overlap,
        apply_clahe=not args.no_clahe,
    )
