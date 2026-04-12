"""
geospatial_utils.py — Geospatial Preprocessing & Inference Stitching
=====================================================================
Principal Engineer: CloudShadow-UNet Project
─────────────────────────────────────────────────────────────────────────────
Implements every geospatial operation in the pipeline:

  PREPROCESSING
  ─────────────
  • read_scene()          Read a 4-band 16-bit GeoTIFF → normalised float32
  • apply_clahe()         Per-band CLAHE contrast enhancement for thin cirrus
  • generate_tile_coords()Boundary-safe sliding window tile coordinates
  • extract_patches()     Slice image + mask into overlapping NumPy arrays
  • save_patches()        Persist arrays as compressed .npy files
  • preprocess_scene()    Full pipeline: read → CLAHE → tile → save

  INFERENCE STITCHING
  ───────────────────
  • cosine_bell_mask()    2-D Hanning-window blending weight mask
  • stitch_predictions()  Accumulate + blend overlapping tile predictions
  • write_mask_geotiff()  Write georeferenced GeoTIFF preserving CRS exactly

  ANALYTICS
  ─────────
  • compute_area_stats()  Per-class area in km² using affine pixel size
  • generate_rgb_preview()Gamma-corrected RGB uint8 image for display

Critical invariant — CRS preservation:
  Every function that writes a GeoTIFF copies source_profile verbatim.
  The affine transform (pixel ↔ world coordinate mapping) and CRS are
  NEVER recomputed or estimated — they are always inherited from the
  source rasterio dataset.  This guarantees pixel-perfect alignment in
  QGIS, ArcGIS, and any standard GIS software.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)

# ─── physical constants ───────────────────────────────────────────────────────
REFLECTANCE_SCALE: float = 10_000.0
"""Sentinel-2 Level-2A and Landsat 8 Level-2 store surface reflectance
as UInt16 integers scaled by 10,000.  Divide by this value to get
reflectance in [0, 1]."""

# CLAHE parameters — tuned for thin cirrus visibility in NIR/SWIR bands
CLAHE_CLIP_LIMIT:  float         = 2.0
CLAHE_TILE_GRID:   tuple[int, int] = (8, 8)

# Class label encoding
CLASS_LABELS: dict[int, str] = {
    0: "Background",
    1: "Cloud",
    2: "Shadow",
}

# Display colour palette (R, G, B) for uint8 mask PNG
CLASS_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    0: (128, 128, 128),   # grey
    1: (255, 255, 255),   # white
    2: (30,  60,  120),   # dark blue
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — READING & NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def read_scene(
    path: Path,
    band_indices: tuple[int, ...] = (1, 2, 3, 4),
) -> tuple[np.ndarray, dict]:
    """Read a multi-band GeoTIFF and return a normalised float32 array.

    Reads the specified band indices (1-based, rasterio convention),
    divides raw UInt16 DN values by REFLECTANCE_SCALE, and clips to [0, 1]
    to handle rare sensor saturation artefacts (DN > 10000).

    Args:
        path:         Absolute path to the source GeoTIFF.
        band_indices: 1-based band indices to read (default: 1,2,3,4 = RGBNIR).

    Returns:
        image:   float32 ndarray of shape (H, W, len(band_indices)), range [0,1].
        profile: rasterio dataset profile dict — contains CRS, transform, dtype,
                 width, height, etc.  Pass this to all downstream write calls.

    Raises:
        ValueError: If the file has fewer bands than requested.
        rasterio.errors.RasterioIOError: If the file cannot be opened.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        if src.count < max(band_indices):
            raise ValueError(
                f"'{path.name}' has {src.count} band(s); "
                f"requested indices {band_indices} require at least {max(band_indices)}."
            )
        raw: np.ndarray = src.read(
            list(band_indices),
            out_dtype=np.float32,
            resampling=Resampling.bilinear,
        )  # shape: (len(band_indices), H, W)
        profile = src.profile.copy()

    # rasterio: (bands, H, W)  →  NumPy-friendly: (H, W, bands)
    image = np.transpose(raw, (1, 2, 0))
    image = np.clip(image / REFLECTANCE_SCALE, 0.0, 1.0, out=image)

    logger.info(
        "Scene loaded: '%s'  shape=%s  CRS=%s  min=%.4f  max=%.4f",
        path.name, image.shape, profile.get("crs"), image.min(), image.max(),
    )
    return image, profile


def read_mask_scene(path: Path) -> np.ndarray:
    """Read a single-band integer label GeoTIFF → uint8 (H, W) array.

    Args:
        path: Path to the ground-truth mask GeoTIFF.

    Returns:
        uint8 ndarray with values in {0, 1, 2}.
    """
    with rasterio.open(path) as src:
        mask: np.ndarray = src.read(1).astype(np.uint8)
    logger.info(
        "Mask loaded: '%s'  shape=%s  unique_labels=%s",
        path.name, mask.shape, sorted(np.unique(mask).tolist()),
    )
    return mask


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — CLAHE CONTRAST ENHANCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid:  tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """Apply per-band CLAHE contrast enhancement to expose thin cirrus clouds.

    Algorithm per band:
      1. Scale float32 [0,1] → uint16 [0,65535]  (CLAHE requires integer input)
      2. Apply cv2.createCLAHE()  (Contrast Limited Adaptive Histogram
         Equalisation, Zuiderveld 1994)
      3. Scale back to float32 [0,1]

    Why CLAHE over global histogram equalisation?
      Global equalisation amplifies noise in bright regions (saturated cloud
      tops) and loses detail in dark regions (cloud shadows).  CLAHE computes
      equalisation over small local tiles and clips the redistribution at
      clip_limit to avoid over-amplification.

    Args:
        image:      float32 (H, W, C) array in [0, 1].
        clip_limit: Maximum contrast amplification per tile (default 2.0).
        tile_grid:  (rows, cols) of CLAHE tiles (default (8, 8)).

    Returns:
        Enhanced float32 (H, W, C) array in [0, 1].
    """
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced  = np.empty_like(image)

    for b in range(image.shape[2]):
        band_u16              = (image[:, :, b] * 65535.0).astype(np.uint16)
        band_eq               = clahe_obj.apply(band_u16)
        enhanced[:, :, b]     = band_eq.astype(np.float32) / 65535.0

    return enhanced


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — SPATIAL TILING
# ═══════════════════════════════════════════════════════════════════════════════

def generate_tile_coords(
    height: int,
    width:  int,
    patch_size: int  = 256,
    overlap:    float = 0.25,
) -> list[tuple[int, int, int, int]]:
    """Generate (r0, r1, c0, c1) tile window coordinates.

    Boundary behaviour:
      The final row and column tile in each direction are shifted
      left/up so they remain full-sized (patch_size × patch_size)
      even when the image dimensions are not exact multiples of the
      stride.  This avoids partial tiles without discarding any pixels.

    Args:
        height:     Image height in pixels.
        width:      Image width in pixels.
        patch_size: Square tile side length.
        overlap:    Fractional overlap in [0, 1).  0.25 = 25 % overlap.

    Returns:
        List of (r0, r1, c0, c1) integer tuples.

    Raises:
        ValueError: If patch_size > min(height, width) or overlap is invalid.
    """
    if patch_size > min(height, width):
        raise ValueError(
            f"patch_size={patch_size} exceeds image dimensions ({height}×{width})."
        )
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1); got {overlap}.")

    stride = max(1, int(patch_size * (1.0 - overlap)))
    coords: list[tuple[int, int, int, int]] = []

    row = 0
    while True:
        r0 = min(row, height - patch_size)
        r1 = r0 + patch_size
        col = 0
        while True:
            c0 = min(col, width - patch_size)
            c1 = c0 + patch_size
            coords.append((r0, r1, c0, c1))
            if col + stride >= width:
                break
            col += stride
        if row + stride >= height:
            break
        row += stride

    logger.debug(
        "Tile coords generated: %d tiles (patch=%d, overlap=%.0f%%)",
        len(coords), patch_size, overlap * 100,
    )
    return coords


def extract_patches(
    image: np.ndarray,
    mask:  Optional[np.ndarray],
    patch_size: int  = 256,
    overlap:    float = 0.25,
) -> tuple[list[np.ndarray], Optional[list[np.ndarray]]]:
    """Slice a scene into overlapping patches, optionally with a mask.

    Args:
        image:      float32 (H, W, 4) normalised scene.
        mask:       uint8   (H, W)    label scene (None for inference).
        patch_size: Tile spatial dimension.
        overlap:    Fractional overlap between adjacent tiles.

    Returns:
        (img_patches, mask_patches) where mask_patches is None if mask is None.
    """
    h, w = image.shape[:2]
    coords = generate_tile_coords(h, w, patch_size=patch_size, overlap=overlap)

    img_patches  = [image[r0:r1, c0:c1, :] for r0, r1, c0, c1 in coords]
    mask_patches = None
    if mask is not None:
        mask_patches = [mask[r0:r1, c0:c1] for r0, r1, c0, c1 in coords]

    logger.info(
        "Extracted %d patches of %d×%d (overlap=%.0f%%)",
        len(img_patches), patch_size, patch_size, overlap * 100,
    )
    return img_patches, mask_patches


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — PATCH PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

def save_patches(
    img_patches:   list[np.ndarray],
    mask_patches:  Optional[list[np.ndarray]],
    out_img_dir:   Path,
    out_mask_dir:  Path,
    scene_stem:    str,
) -> int:
    """Save image and mask patches as .npy files.

    Files are named:  {scene_stem}_patch_{idx:05d}.npy
    Existing files with the same names are silently overwritten.

    Args:
        img_patches:  List of float32 (H, W, 4) arrays.
        mask_patches: List of uint8  (H, W)    arrays (may be None).
        out_img_dir:  Destination directory for image patches.
        out_mask_dir: Destination directory for mask patches.
        scene_stem:   Base name prefix (typically the GeoTIFF filename stem).

    Returns:
        Number of image patches saved.
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    for idx, patch in enumerate(img_patches):
        np.save(out_img_dir / f"{scene_stem}_patch_{idx:05d}.npy", patch)

    if mask_patches is not None:
        for idx, mpatch in enumerate(mask_patches):
            np.save(out_mask_dir / f"{scene_stem}_patch_{idx:05d}.npy", mpatch)

    logger.info(
        "Saved %d patches → '%s'", len(img_patches), out_img_dir
    )
    return len(img_patches)


def preprocess_scene(
    image_path: Path,
    mask_path:  Optional[Path],
    out_img_dir:  Path,
    out_mask_dir: Path,
    patch_size: int   = 256,
    overlap:    float = 0.25,
    enhance:    bool  = True,
) -> int:
    """Full preprocessing pipeline for a single GeoTIFF scene.

    Steps:
      1. Read 4-band GeoTIFF → float32 [0,1]
      2. (Optional) Per-band CLAHE enhancement
      3. (Optional) Read corresponding ground-truth mask
      4. Tile image (+ mask) into overlapping 256×256 patches
      5. Save all patches as .npy files

    Args:
        image_path:   Source 4-band GeoTIFF.
        mask_path:    Ground-truth mask GeoTIFF (None for inference scenes).
        out_img_dir:  Output directory for image patches.
        out_mask_dir: Output directory for mask patches.
        patch_size:   Tile side length.
        overlap:      Fractional tile overlap.
        enhance:      Apply CLAHE enhancement (default True).

    Returns:
        Number of patches created.
    """
    image, _ = read_scene(image_path)

    if enhance:
        logger.info("Applying CLAHE enhancement …")
        image = apply_clahe(image)

    mask = read_mask_scene(mask_path) if mask_path is not None else None
    img_patches, mask_patches = extract_patches(image, mask, patch_size, overlap)

    return save_patches(
        img_patches,
        mask_patches,
        out_img_dir  = out_img_dir,
        out_mask_dir = out_mask_dir,
        scene_stem   = Path(image_path).stem,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — COSINE-BELL BLENDING & STITCHING
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_bell_mask(patch_size: int) -> np.ndarray:
    """Generate a 2-D Hanning-window (cosine-bell) blending weight mask.

    The Hanning window tapers smoothly from 1.0 at the centre to 0.0 at
    the edges using a raised cosine:
        w(n) = 0.5 × (1 − cos(2πn / (N−1)))

    Taking the outer product of two 1-D Hanning windows gives a 2-D bell
    that peaks at the patch centre.  When overlapping tiles are blended
    with these weights:
      • Centre pixels (most context, most reliable) contribute maximally.
      • Edge pixels (truncated context, artefact-prone) contribute minimally.
    Result: seamless predictions with no visible grid lines.

    Args:
        patch_size: Side length of the square patch.

    Returns:
        float64 (patch_size, patch_size) weight array, values ∈ (0, 1].
    """
    window_1d = np.hanning(patch_size)
    window_2d = np.outer(window_1d, window_1d)
    window_2d = window_2d / window_2d.max()   # normalise peak to 1.0
    return window_2d.astype(np.float64)


def stitch_predictions(
    model,
    image:      np.ndarray,
    patch_size: int   = 256,
    overlap:    float = 0.25,
    batch_size: int   = 8,
    num_classes: int  = 3,
) -> np.ndarray:
    """Run sliding-window inference and stitch predictions with cosine blending.

    Algorithm:
      For each tile position:
        1. Extract patch from image.
        2. Stack up to batch_size patches → call model.predict().
        3. Multiply each tile's softmax output by the cosine-bell weight mask.
        4. Accumulate into a full-scene probability array.
        5. Accumulate the weight mask into a weight sum array.
      After all tiles:
        6. Divide probability accumulator by weight sum → blended probs.
        7. Argmax over class axis → (H, W) int32 class map.

    OOM prevention:
      model.predict() is called once per mini-batch (batch_size tiles at a
      time) rather than once per tile.  This amortises the Python/GPU call
      overhead while keeping GPU memory bounded.

    Args:
        model:       Loaded Keras model with softmax output.
        image:       float32 (H, W, 4) scene array (preprocessed + enhanced).
        patch_size:  Inference tile size.
        overlap:     Tile overlap fraction.
        batch_size:  Number of tiles per model.predict() call.
        num_classes: Number of output segmentation classes.

    Returns:
        int32 (H, W) predicted class label map.
    """
    h, w   = image.shape[:2]
    coords = generate_tile_coords(h, w, patch_size=patch_size, overlap=overlap)
    bell   = cosine_bell_mask(patch_size)

    # Pre-allocate accumulation arrays in float64 to avoid summation error
    prob_accum   = np.zeros((h, w, num_classes), dtype=np.float64)
    weight_accum = np.zeros((h, w),              dtype=np.float64)

    n_total = len(coords)
    logger.info("Stitching %d tiles × %d px — %d classes …", n_total, patch_size, num_classes)

    for batch_start in range(0, n_total, batch_size):
        batch_coords = coords[batch_start: batch_start + batch_size]
        patches = np.stack(
            [image[r0:r1, c0:c1, :] for r0, r1, c0, c1 in batch_coords],
            axis=0,
        ).astype(np.float32)

        # Shape: (mini_bs, patch_size, patch_size, num_classes)
        preds = model.predict(patches, verbose=0)

        for (r0, r1, c0, c1), pred in zip(batch_coords, preds):
            prob_accum  [r0:r1, c0:c1, :] += pred.astype(np.float64) * bell[:, :, np.newaxis]
            weight_accum[r0:r1, c0:c1]    += bell

        processed = min(batch_start + batch_size, n_total)
        if processed % (batch_size * 10) == 0 or processed == n_total:
            logger.info("  Inference progress: %d / %d tiles", processed, n_total)

    # Avoid division by zero (guard for any pixel unvisited by tiles)
    weight_accum = np.where(weight_accum == 0.0, 1.0, weight_accum)
    blended      = prob_accum / weight_accum[:, :, np.newaxis]
    class_map    = np.argmax(blended, axis=-1).astype(np.int32)

    # Log final class distribution
    total = class_map.size
    for cls_id, name in CLASS_LABELS.items():
        cnt = int(np.sum(class_map == cls_id))
        logger.info("  %s: %d px (%.2f %%)", name, cnt, 100.0 * cnt / total)

    return class_map


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — GEOREFERENCED OUTPUT WRITER
# ═══════════════════════════════════════════════════════════════════════════════

def write_mask_geotiff(
    class_map:      np.ndarray,
    source_profile: dict,
    output_path:    Path,
) -> None:
    """Write a predicted class map as a georeferenced GeoTIFF.

    CRS invariant:
      source_profile["crs"]       → written verbatim to output
      source_profile["transform"] → written verbatim to output
      → The output mask aligns pixel-for-pixel with source imagery in QGIS.

    Encoding:
      Band 1, uint8: 0 = Background, 1 = Cloud, 2 = Shadow, 255 = NoData

    Compression:
      LZW with horizontal differencing (predictor=2) — lossless, effective
      for near-constant integer masks.  Typical ratio: 10:1 to 30:1.

    Args:
        class_map:      int32 (H, W) predicted label array.
        source_profile: rasterio profile dict from the source GeoTIFF.
        output_path:    Destination GeoTIFF path.

    Raises:
        rasterio.errors.RasterioIOError: If the output directory is not writable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = source_profile.copy()
    profile.update({
        "count":    1,
        "dtype":    "uint8",
        "driver":   "GTiff",
        "compress": "lzw",
        "predictor": 2,      # Horizontal differencing predictor for LZW
        "nodata":   255,
        "tiled":    True,    # Tiled GeoTIFF for efficient random access
        "blockxsize": 256,
        "blockysize": 256,
        "interleave": "band",
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(class_map.astype(np.uint8), 1)
        # Write colourmap for automatic styling in QGIS
        colormap = {
            cls_id: (*rgb, 255)   # RGBA — alpha=255 (fully opaque)
            for cls_id, rgb in CLASS_COLORS_RGB.items()
        }
        colormap[255] = (0, 0, 0, 0)   # NoData = transparent
        dst.write_colormap(1, colormap)

    logger.info(
        "Mask GeoTIFF written → %s\n  CRS=%s\n  Transform=%s",
        output_path,
        profile.get("crs"),
        profile.get("transform"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — AREA STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_area_stats(
    class_map:      np.ndarray,
    source_profile: dict,
) -> dict[str, float]:
    """Compute per-class pixel counts and area in km².

    The ground sampling distance (GSD) is derived from the diagonal of the
    affine transform matrix:
        pixel_width_m  = |transform.a|
        pixel_height_m = |transform.e|

    For Sentinel-2 B02–B08 (10 m GSD): pixel_area = 100 m² = 0.0001 km²
    For Landsat 8 OLI (30 m GSD): pixel_area = 900 m² = 0.0009 km²

    Args:
        class_map:      (H, W) integer class label array.
        source_profile: rasterio profile dict with 'transform' key.

    Returns:
        Dictionary with keys:
          background_px, cloud_px, shadow_px,
          background_km2, cloud_km2, shadow_km2,
          total_km2, cloud_fraction, shadow_fraction
    """
    transform = source_profile.get("transform")
    stats: dict[str, float] = {}

    # Pixel counts
    for cls_id, name in CLASS_LABELS.items():
        cnt = int(np.sum(class_map == cls_id))
        stats[f"{name.lower()}_px"] = float(cnt)

    stats["total_px"] = float(class_map.size)

    if transform is None:
        logger.warning(
            "Source profile has no affine transform — km² stats unavailable."
        )
        return stats

    pixel_w_m   = abs(transform.a)
    pixel_h_m   = abs(transform.e)
    pixel_area  = pixel_w_m * pixel_h_m          # m²
    to_km2      = pixel_area / 1_000_000.0       # m² → km²

    for cls_id, name in CLASS_LABELS.items():
        px_key  = f"{name.lower()}_px"
        km2_key = f"{name.lower()}_km2"
        stats[km2_key] = round(stats[px_key] * to_km2, 4)

    stats["total_km2"]        = round(stats["total_px"] * to_km2, 4)
    stats["pixel_size_m"]     = round(pixel_w_m, 2)
    stats["cloud_fraction"]   = round(
        stats.get("cloud_km2", 0) / max(stats["total_km2"], 1e-9), 6
    )
    stats["shadow_fraction"]  = round(
        stats.get("shadow_km2", 0) / max(stats["total_km2"], 1e-9), 6
    )

    logger.info(
        "Area stats — GSD=%.1f m  | Cloud=%.2f km² (%.1f %%)  | Shadow=%.2f km² (%.1f %%)",
        pixel_w_m,
        stats.get("cloud_km2", 0),  stats.get("cloud_fraction", 0) * 100,
        stats.get("shadow_km2", 0), stats.get("shadow_fraction", 0) * 100,
    )
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — RGB PREVIEW GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_rgb_preview(image: np.ndarray, gamma: float = 0.50) -> np.ndarray:
    """Extract RGB bands from a 4-channel image for display.

    Sentinel-2 imagery is often dark in linear reflectance space —
    especially over water and shadow.  A gamma of 0.5 (square root)
    approximately matches the perceptual brightness of a standard
    JPEG photograph without clipping bright cloud tops.

    Args:
        image: float32 (H, W, 4) normalised RGBNIR image.
        gamma: Display gamma exponent (< 1 = brightens; default 0.5).

    Returns:
        uint8 (H, W, 3) gamma-corrected RGB display image.
    """
    rgb = image[:, :, :3]
    rgb = np.power(np.clip(rgb, 0.0, 1.0), gamma)
    return (rgb * 255.0).astype(np.uint8)


def class_map_to_rgb(class_map: np.ndarray) -> np.ndarray:
    """Convert a (H, W) label array to a (H, W, 3) colour-coded uint8 image.

    Args:
        class_map: int32 (H, W) array with values {0, 1, 2}.

    Returns:
        uint8 (H, W, 3) RGB colour mask suitable for display or export.
    """
    rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_id, colour in CLASS_COLORS_RGB.items():
        rgb[class_map == cls_id] = colour
    return rgb
