"""
scripts/download_38cloud.py
============================
Automated downloader and converter for the 38-Cloud Dataset.

The 38-Cloud dataset (Mohajerani & Saeedi, 2019) is the standard benchmark
for satellite cloud segmentation.  It contains:
  • 38 Landsat 8 OLI scenes (30 m GSD)
  • 4 separate band files per scene: red, green, blue, nir  (GeoTIFF, uint16)
  • 1 ground-truth binary cloud mask per scene (PNG, 0=clear / 255=cloud)

This script:
  1. Downloads the dataset from Kaggle (requires kaggle API credentials)
     OR accepts a manually downloaded zip file path.
  2. Merges the 4 separate band files into a single 4-band GeoTIFF per scene.
  3. Converts the binary PNG masks → 3-class GeoTIFF masks
     (0=Background, 1=Cloud, 2=Shadow — shadow is estimated here; see note).
  4. Saves everything under data/raw/ in the format expected by preprocess.py.

Note on Shadow Labels:
  The 38-Cloud dataset provides only binary cloud masks (cloud / clear).
  Shadow labels (class 2) are NOT included in 38-Cloud.
  Options:
    A. Use 38-Cloud as-is with 2 classes (background + cloud) — set NUM_CLASSES=2.
    B. Use the 95-Cloud dataset which has shadow annotations.
    C. Use the Sentinel-2 Cloud Shadow dataset from Roboflow (see below).

Dataset sources:
  38-Cloud (Kaggle):  https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images
  95-Cloud (Kaggle):  https://www.kaggle.com/datasets/sorour/95cloud-cloud-segmentation-on-small-patched-images
  Sentinel-2 (Roboflow): https://universe.roboflow.com/cloud-detection/cloud-shadow-segmentation

Usage:
  # Using Kaggle API (automatic download):
  python scripts/download_38cloud.py --source kaggle --dataset 38cloud

  # Using local zip (already downloaded manually):
  python scripts/download_38cloud.py --source local --zip_path ~/Downloads/38cloud.zip

  # Use 95-Cloud for cloud+shadow labels:
  python scripts/download_38cloud.py --source kaggle --dataset 95cloud
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"

# ─── Kaggle dataset identifiers ───────────────────────────────────────────────
KAGGLE_DATASETS = {
    "38cloud": "sorour/38cloud-cloud-segmentation-in-satellite-images",
    "95cloud": "sorour/95cloud-cloud-segmentation-on-small-patched-images",
}


# ─── step 1: download via Kaggle API ──────────────────────────────────────────

def download_from_kaggle(dataset_key: str, dest_dir: Path) -> Path:
    """Download a Kaggle dataset using the kaggle CLI.

    Requires:
      pip install kaggle
      ~/.kaggle/kaggle.json  (or KAGGLE_USERNAME + KAGGLE_KEY env vars)

    Args:
        dataset_key: Key in KAGGLE_DATASETS dict ('38cloud' or '95cloud').
        dest_dir:    Where to save the downloaded zip.

    Returns:
        Path to the downloaded zip file.

    Raises:
        SystemExit: If kaggle CLI is not installed or credentials are missing.
    """
    if shutil.which("kaggle") is None:
        logger.error(
            "kaggle CLI not found.\n"
            "Run: pip install kaggle\n"
            "Then set up credentials: https://www.kaggle.com/docs/api"
        )
        sys.exit(1)

    dataset_id = KAGGLE_DATASETS.get(dataset_key)
    if dataset_id is None:
        logger.error("Unknown dataset key '%s'. Choose from: %s", dataset_key, list(KAGGLE_DATASETS))
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading '%s' from Kaggle …", dataset_id)

    cmd = [
        "kaggle", "datasets", "download",
        "--dataset", dataset_id,
        "--path", str(dest_dir),
        "--unzip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Kaggle download failed:\n%s", result.stderr)
        sys.exit(1)

    logger.info("Download complete → %s", dest_dir)
    return dest_dir


# ─── step 2: merge 4 band files → single GeoTIFF ─────────────────────────────

def merge_bands_to_geotiff(
    band_paths: dict[str, Path],
    output_path: Path,
) -> None:
    """Stack 4 single-band GeoTIFFs into one 4-band GeoTIFF.

    Args:
        band_paths: Dict mapping {'red': Path, 'green': Path, 'blue': Path, 'nir': Path}.
        output_path: Destination 4-band GeoTIFF path.
    """
    import rasterio

    band_order = ["red", "green", "blue", "nir"]
    bands: list[np.ndarray] = []
    profile = None

    for band_name in band_order:
        path = band_paths.get(band_name)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"Band file not found for '{band_name}': {path}"
            )
        with rasterio.open(path) as src:
            bands.append(src.read(1))
            if profile is None:
                profile = src.profile.copy()

    profile.update({
        "count": 4,
        "dtype": "uint16",
        "driver": "GTiff",
        "compress": "lzw",
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band, i)

    logger.debug("Merged 4-band GeoTIFF → %s", output_path)


# ─── step 3: convert mask PNG → 3-class GeoTIFF ──────────────────────────────

def convert_mask_png_to_geotiff(
    mask_png_path: Path,
    reference_tif_path: Path,
    output_path: Path,
    dataset_type: str = "38cloud",
) -> None:
    """Convert a PNG cloud mask to a 3-class uint8 GeoTIFF.

    38-Cloud mask encoding:
        0   → clear sky  (Background, class 0)
        255 → cloud      (Cloud, class 1)
        (no shadow labels — all non-cloud → Background)

    95-Cloud mask encoding (if available):
        0   → clear sky
        64  → cloud shadow (class 2)
        255 → cloud

    The output GeoTIFF copies the CRS and affine transform from
    reference_tif_path so the mask is georeferenced identically.

    Args:
        mask_png_path:      Path to the PNG mask.
        reference_tif_path: Path to the corresponding 4-band GeoTIFF
                            (used to copy CRS + transform).
        output_path:        Destination georeferenced mask GeoTIFF.
        dataset_type:       '38cloud' or '95cloud' — controls label mapping.
    """
    import cv2
    import rasterio

    png = cv2.imread(str(mask_png_path), cv2.IMREAD_GRAYSCALE)
    if png is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_png_path}")

    # Build 3-class label map
    label_map = np.zeros_like(png, dtype=np.uint8)  # 0 = Background

    if dataset_type == "95cloud":
        label_map[png == 64]  = 2   # Shadow
        label_map[png == 255] = 1   # Cloud
    else:
        # 38-Cloud: binary only
        label_map[png >= 128] = 1   # Cloud

    # Copy CRS and affine transform from reference GeoTIFF
    with rasterio.open(reference_tif_path) as ref:
        profile = ref.profile.copy()

    profile.update({
        "count":  1,
        "dtype":  "uint8",
        "driver": "GTiff",
        "compress": "lzw",
        "nodata": 255,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(label_map, 1)

    logger.debug("Mask GeoTIFF → %s  unique=%s", output_path, np.unique(label_map).tolist())


# ─── step 4: full conversion pipeline ────────────────────────────────────────

def convert_38cloud(source_dir: Path, dataset_type: str = "38cloud") -> int:
    """Convert raw 38-Cloud / 95-Cloud directory structure to project format.

    38-Cloud directory structure after download:
        38-Cloud_training/
            train_red/     *.TIF
            train_green/   *.TIF
            train_blue/    *.TIF
            train_nir/     *.TIF
            train_gt/      *.TIF  (or *.png)

    Output written to data/raw/:
        scene_<stem>.tif          ← 4-band merged GeoTIFF
        scene_<stem>_mask.tif     ← 3-class georeferenced mask

    Args:
        source_dir:   Root directory of the downloaded dataset.
        dataset_type: '38cloud' or '95cloud'.

    Returns:
        Number of scenes successfully converted.
    """
    logger.info("Scanning '%s' for %s data …", source_dir, dataset_type)

    # Locate band directories (handle various sub-folder naming conventions)
    def _find_dir(root: Path, keywords: list[str]) -> Path | None:
        for kw in keywords:
            matches = list(root.rglob(f"*{kw}*"))
            dirs = [m for m in matches if m.is_dir()]
            if dirs:
                return dirs[0]
        return None

    red_dir   = _find_dir(source_dir, ["red",   "Red"])
    green_dir = _find_dir(source_dir, ["green", "Green"])
    blue_dir  = _find_dir(source_dir, ["blue",  "Blue"])
    nir_dir   = _find_dir(source_dir, ["nir",   "NIR"])
    gt_dir    = _find_dir(source_dir, ["gt", "GT", "mask", "label"])

    if not all([red_dir, green_dir, blue_dir, nir_dir, gt_dir]):
        logger.error(
            "Could not find all required band directories.\n"
            "Expected: red, green, blue, nir, gt  under '%s'.\n"
            "Found: red=%s green=%s blue=%s nir=%s gt=%s",
            source_dir, red_dir, green_dir, blue_dir, nir_dir, gt_dir,
        )
        return 0

    # Collect all red-band files and match other bands by stem
    red_files = sorted(list(red_dir.glob("*.TIF")) + list(red_dir.glob("*.tif")))
    if not red_files:
        logger.error("No .TIF files found in red band directory: %s", red_dir)
        return 0

    logger.info("Found %d scenes to convert.", len(red_files))
    count = 0

    for red_path in red_files:
        stem = red_path.stem.replace("_red", "").replace("_RED", "").replace("red_", "")

        # Find matching files for other bands and mask
        def _find_file(directory: Path, pattern: str) -> Path | None:
            candidates = (
                list(directory.glob(f"*{stem}*.TIF")) +
                list(directory.glob(f"*{stem}*.tif")) +
                list(directory.glob(f"*{stem}*.png")) +
                list(directory.glob(f"*{stem}*.PNG"))
            )
            return candidates[0] if candidates else None

        green_path = _find_file(green_dir, stem)
        blue_path  = _find_file(blue_dir,  stem)
        nir_path   = _find_file(nir_dir,   stem)
        gt_path    = _find_file(gt_dir,    stem)

        if not all([green_path, blue_path, nir_path, gt_path]):
            logger.warning(
                "Skipping '%s' — missing files: green=%s blue=%s nir=%s gt=%s",
                stem, green_path, blue_path, nir_path, gt_path,
            )
            continue

        out_image = RAW_DIR / f"scene_{stem}.tif"
        out_mask  = RAW_DIR / f"scene_{stem}_mask.tif"

        try:
            merge_bands_to_geotiff(
                band_paths={
                    "red":   red_path,
                    "green": green_path,
                    "blue":  blue_path,
                    "nir":   nir_path,
                },
                output_path=out_image,
            )
            convert_mask_png_to_geotiff(
                mask_png_path      = gt_path,
                reference_tif_path = out_image,
                output_path        = out_mask,
                dataset_type       = dataset_type,
            )
            count += 1
            logger.info("[%d/%d] ✅ %s", count, len(red_files), out_image.name)

        except Exception as exc:
            logger.error("Failed to convert scene '%s': %s", stem, exc)

    logger.info("Conversion complete: %d / %d scenes → data/raw/", count, len(red_files))
    return count


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and convert 38-Cloud / 95-Cloud dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["kaggle", "local"],
        default="kaggle",
        help="'kaggle' = auto-download via API; 'local' = use existing zip/folder",
    )
    parser.add_argument(
        "--dataset",
        choices=["38cloud", "95cloud"],
        default="38cloud",
        help="Which dataset to download (95cloud has shadow labels)",
    )
    parser.add_argument(
        "--zip_path",
        type=Path,
        default=None,
        help="[local mode] Path to manually downloaded zip or extracted folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.source == "kaggle":
        dl_dir = PROJECT_ROOT / "data" / "_download_temp"
        download_from_kaggle(args.dataset, dl_dir)
        convert_38cloud(dl_dir, dataset_type=args.dataset)
        shutil.rmtree(dl_dir, ignore_errors=True)

    elif args.source == "local":
        if args.zip_path is None:
            logger.error("--zip_path is required when --source=local")
            sys.exit(1)
        zip_path = Path(args.zip_path)
        if not zip_path.exists():
            logger.error("Path not found: %s", zip_path)
            sys.exit(1)

        if zip_path.is_file() and zip_path.suffix == ".zip":
            extract_dir = PROJECT_ROOT / "data" / "_extract_temp"
            logger.info("Extracting %s …", zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
            convert_38cloud(extract_dir, dataset_type=args.dataset)
            shutil.rmtree(extract_dir, ignore_errors=True)
        elif zip_path.is_dir():
            convert_38cloud(zip_path, dataset_type=args.dataset)
        else:
            logger.error("zip_path must be a .zip file or a directory.")
            sys.exit(1)
