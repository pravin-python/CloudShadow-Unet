"""
Module 5 — Geospatial Post-Processing & Re-stitching
=====================================================
Runs inference on a full-scene 4-band GeoTIFF using a sliding-window
strategy.  Overlapping prediction tiles are blended via a cosine-bell
weight mask to suppress grid artefacts at tile boundaries.

The predicted class map (argmax of softmax output) is written back to
disk as a GeoTIFF with the exact CRS, affine transform, and bounding box
copied from the source raster — ensuring pixel-perfect alignment in QGIS.

Usage (CLI):
    python src/inference/predict.py \
        --input  data/raw/scene.tif \
        --output outputs/predicted_mask.tif \
        --model  models/best_weights.h5 \
        --patch_size 256 \
        --overlap 0.25

Algorithm
---------
1.  Read source GeoTIFF → (H, W, 4) float32 array  [+ preserve profile]
2.  Apply CLAHE (same as training)
3.  Generate patch coordinates with 25% overlap
4.  For each patch:
        a. Extract sub-array from the padded image
        b. Run model.predict(patch[np.newaxis])  → (1, P, P, 3)
        c. Multiply softmax maps by cosine-bell weight mask
        d. Accumulate into (H, W, 3) sum array and (H, W) weight array
5.  Divide accumulated sum by weight sum → blended probability map
6.  Argmax over class axis → (H, W) int32 class map
7.  Write output GeoTIFF with source spatial metadata preserved
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path to resolve 'src' module
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
 
import numpy as np
import rasterio

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# Label colour palette for optional preview PNG
# (R, G, B): 0=Background grey, 1=Cloud white, 2=Shadow dark-blue
CLASS_COLORS = {0: (128, 128, 128), 1: (255, 255, 255), 2: (30, 60, 120)}


# ─── cosine-bell weight mask ──────────────────────────────────────────────────

def _cosine_bell_mask(patch_size: int) -> np.ndarray:
    """Generate a 2-D cosine-bell weight mask of shape (P, P).

    Weights are highest at the patch centre (1.0) and taper smoothly
    to 0 at the patch edges.  This ensures that edge pixels, which are
    most affected by context truncation, contribute minimally to the
    final blend, while centre pixels (most reliable predictions) dominate.

    Args:
        patch_size: Side length of the square patch.

    Returns:
        float32 (patch_size, patch_size) weight array in (0, 1].
    """
    window_1d = np.hanning(patch_size).astype(np.float32)
    window_2d = np.outer(window_1d, window_1d)
    # Hanning window already tapers to 0 at edges; normalise peak to 1
    window_2d /= window_2d.max()
    # Add a tiny epsilon to prevent division by zero in areas with minimal overlap.
    return window_2d + 1e-6


# ─── image I/O ────────────────────────────────────────────────────────────────

def _read_source(path: Path) -> tuple[np.ndarray, dict]:
    """Read a 4-band GeoTIFF and return normalised float32 array + profile."""
    from src.preprocessing.preprocess import (
        REFLECTANCE_SCALE,
        apply_clahe_per_band,
    )

    with rasterio.open(path) as src:
        if src.count < 4:
            raise ValueError(f"{path} has {src.count} bands; need 4 (R,G,B,NIR)")
        raw = src.read([1, 2, 3, 4], out_dtype=np.float32)
        profile = src.profile.copy()

    image = np.transpose(raw, (1, 2, 0))
    image = np.clip(image / REFLECTANCE_SCALE, 0.0, 1.0)
    image = apply_clahe_per_band(image)
    return image, profile


# ─── sliding window inference ─────────────────────────────────────────────────

def sliding_window_predict(
    models: list,
    image: np.ndarray,
    patch_size: int = 256,
    overlap: float = 0.25,
    batch_size: int = 8,
) -> np.ndarray:
    """Run model inference over a full-scene image with overlapping tiles.

    Args:
        models:     List of loaded Keras models with softmax output (H, W, num_classes).
        image:      float32 (H, W, 4) normalised scene array.
        patch_size: Spatial size of each inference tile.
        overlap:    Fractional overlap between adjacent tiles.
        batch_size: Number of tiles processed per model.predict() call.

    Returns:
        int32 (H, W) predicted class label map.
    """
    from src.preprocessing.preprocess import generate_patch_coords

    h, w = image.shape[:2]
    num_classes = models[0].output_shape[-1]

    # Accumulation arrays for weighted blending
    prob_accum = np.zeros((h, w, num_classes), dtype=np.float64)
    weight_accum = np.zeros((h, w), dtype=np.float64)

    bell = _cosine_bell_mask(patch_size).astype(np.float64)

    coords = generate_patch_coords(h, w, patch_size=patch_size, overlap=overlap)
    n_patches = len(coords)
    logger.info("Running inference on %d patches …", n_patches)

    # Process in mini-batches to saturate GPU throughput
    for batch_start in range(0, n_patches, batch_size):
        batch_coords = coords[batch_start: batch_start + batch_size]
        patches = np.stack(
            [image[r0:r1, c0:c1, :] for r0, r1, c0, c1 in batch_coords],
            axis=0,
        ).astype(np.float32)

        # Shape: (mini_batch, H, W, num_classes)
        ensemble_preds = [m.predict(patches, verbose=0) for m in models]
        preds = np.mean(ensemble_preds, axis=0)

        for (r0, r1, c0, c1), pred in zip(batch_coords, preds):
            # pred: (patch_size, patch_size, num_classes) float32
            prob_accum[r0:r1, c0:c1, :] += pred * bell[:, :, np.newaxis]
            weight_accum[r0:r1, c0:c1] += bell

        if (batch_start // batch_size) % 10 == 0:
            logger.info(
                "  Processed %d / %d patches",
                min(batch_start + batch_size, n_patches),
                n_patches,
            )

    # Avoid division by zero for any unvisited pixel (shouldn't happen with
    # the coord generator but guards against edge cases at image boundaries)
    weight_accum = np.where(weight_accum == 0, 1.0, weight_accum)

    blended_probs = prob_accum / weight_accum[:, :, np.newaxis]
    class_map = np.argmax(blended_probs, axis=-1).astype(np.int32)

    unique, counts = np.unique(class_map, return_counts=True)
    total = class_map.size
    for cls_id, cnt in zip(unique, counts):
        labels = {0: "Background", 1: "Cloud", 2: "Shadow"}
        logger.info(
            "  Class %d (%s): %d px — %.2f %%",
            cls_id, labels.get(cls_id, "Unknown"), cnt, 100.0 * cnt / total,
        )

    return class_map


# ─── geospatial output writer ─────────────────────────────────────────────────

def write_predicted_mask(
    class_map: np.ndarray,
    source_profile: dict,
    output_path: Path,
) -> None:
    """Write the predicted class map as a GeoTIFF preserving source CRS.

    Critically:
        - The affine transform (source_profile["transform"]) maps pixel
          coordinates to real-world coordinates.  Copying it verbatim
          ensures the output mask overlays the source image pixel-for-pixel
          in QGIS, ArcGIS, or any GIS software.
        - The CRS (source_profile["crs"]) ensures reprojection is possible.
        - dtype is set to uint8 to minimise disk footprint.

    Args:
        class_map:      int32 (H, W) predicted label array.
        source_profile: rasterio profile dict from the source GeoTIFF.
        output_path:    Destination path for the output GeoTIFF.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = source_profile.copy()
    profile.update(
        {
            "count": 1,
            "dtype": "uint8",
            "compress": "lzw",       # Lossless compression for integer masks
            "predictor": 2,          # Horizontal differencing — effective for masks
            "nodata": 255,           # Reserve 255 as nodata sentinel
            "driver": "GTiff",
        }
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(class_map.astype(np.uint8), 1)

    logger.info("Predicted mask written → %s", output_path)
    logger.info(
        "CRS: %s | Transform: %s",
        profile.get("crs"),
        profile.get("transform"),
    )


# ─── geospatial statistics ────────────────────────────────────────────────────

def compute_area_statistics(
    class_map: np.ndarray,
    source_profile: dict,
) -> dict[str, float]:
    """Compute per-class area coverage in square kilometres.

    Ground sampling distance (GSD) is derived from the affine transform.
    For Sentinel-2 Band 2-8: GSD ≈ 10 m → pixel area ≈ 100 m².

    Args:
        class_map:      (H, W) integer class label array.
        source_profile: rasterio profile dict containing the affine transform.

    Returns:
        Dictionary mapping class name → area in km².
    """
    transform = source_profile.get("transform")
    if transform is None:
        logger.warning("No affine transform in profile; cannot compute km² areas.")
        return {}

    # Pixel width and height in map units (usually metres for UTM CRS)
    pixel_width_m = abs(transform.a)
    pixel_height_m = abs(transform.e)
    pixel_area_m2 = pixel_width_m * pixel_height_m
    pixel_area_km2 = pixel_area_m2 / 1_000_000.0

    labels = {0: "background_km2", 1: "cloud_km2", 2: "shadow_km2"}
    stats: dict[str, float] = {}
    for cls_id, name in labels.items():
        count = int(np.sum(class_map == cls_id))
        stats[name] = round(count * pixel_area_km2, 4)

    stats["total_scene_km2"] = round(class_map.size * pixel_area_km2, 4)
    stats["cloud_fraction"] = round(
        stats["cloud_km2"] / max(stats["total_scene_km2"], 1e-9), 4
    )
    stats["shadow_fraction"] = round(
        stats["shadow_km2"] / max(stats["total_scene_km2"], 1e-9), 4
    )

    for key, val in stats.items():
        logger.info("  %-25s: %s", key, val)

    return stats


# ─── main inference pipeline ──────────────────────────────────────────────────

def run_inference(
    input_path: Path,
    output_path: Path,
    model_paths: list[Path],
    patch_size: int = 256,
    overlap: float = 0.25,
    batch_size: int = 8,
) -> dict[str, float]:
    """End-to-end inference for a single GeoTIFF scene.

    Args:
        input_path:  Path to 4-band source GeoTIFF.
        output_path: Path for predicted mask GeoTIFF.
        model_paths: List of paths to saved Keras model weights or SavedModel dirs.
        patch_size:  Inference tile size.
        overlap:     Tile overlap fraction.
        batch_size:  Mini-batch size for model.predict().

    Returns:
        Area statistics dictionary (class → km²).
    """
    import tensorflow as tf

    # Load models
    from src.model.losses import CombinedLoss, DiceCoefficient, MeanIoU
    models = []
    for model_path in model_paths:
        logger.info("Loading model from %s …", model_path)
        m = tf.keras.models.load_model(
            str(model_path),
            custom_objects={
                "CombinedLoss": CombinedLoss,
                "DiceCoefficient": DiceCoefficient,
                "MeanIoU": MeanIoU,
            },
        )
        models.append(m)

    # Read & preprocess source scene
    logger.info("Reading source GeoTIFF: %s", input_path)
    image, profile = _read_source(input_path)

    # Sliding-window prediction
    class_map = sliding_window_predict(
        models, image,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
    )

    # Write georeferenced output
    write_predicted_mask(class_map, profile, output_path)

    # Compute area statistics
    stats = compute_area_statistics(class_map, profile)
    return stats


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cloud/shadow inference on a GeoTIFF")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        nargs="+",
        default=[Path(os.environ.get("MODEL_PATH", "models/best_weights.h5"))],
    )
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_inference(
        input_path=args.input,
        output_path=args.output,
        model_paths=args.model,
        patch_size=args.patch_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
