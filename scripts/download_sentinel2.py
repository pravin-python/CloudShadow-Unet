"""
scripts/download_sentinel2.py
==============================
Download Sentinel-2 L2A imagery (10 m GSD) directly from
ESA's Copernicus Data Space Ecosystem using the CDSE OData API.

Sentinel-2 has BOTH clouds and cloud shadows clearly visible —
making it superior to Landsat 8 for cloud shadow segmentation.

What this script downloads:
  • Sentinel-2 Level-2A granules (BOA surface reflectance, 10 m GSD)
  • Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
  • Saves each granule as a 4-band GeoTIFF: scene_<id>.tif
  • For training: you still need to annotate masks manually
    OR use one of the pre-annotated options listed below.

Pre-annotated Sentinel-2 cloud datasets:
  1. CloudSEN12 (Aybar et al., 2022):
       https://huggingface.co/datasets/isp-uv-es/CloudSEN12Plus
       Contains 49,000 Sentinel-2 patches with cloud + shadow labels.
       BEST dataset for this project.

  2. Sentinel-2 Cloud Shadow (Roboflow):
       https://universe.roboflow.com/cloud-detection/cloud-shadow-segmentation

  3. DynamicEarthNet Cloud Mask (IEEE GRSS):
       https://mediatum.ub.tum.de/1650201

Requirements:
  pip install sentinelsat requests tqdm

Usage:
  python scripts/download_sentinel2.py \
      --username YOUR_CDSE_USERNAME \
      --password YOUR_CDSE_PASSWORD \
      --bbox "10.0,45.0,12.0,47.0" \
      --date_start 2024-06-01 \
      --date_end   2024-06-30 \
      --cloud_cover_max 30 \
      --n_scenes 5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"


def _check_dependencies() -> None:
    missing = []
    for pkg in ["requests", "tqdm"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error("Missing packages: %s\nRun: pip install %s", missing, " ".join(missing))
        sys.exit(1)


def search_sentinel2_scenes(
    username: str,
    password: str,
    bbox: tuple[float, float, float, float],  # (lon_min, lat_min, lon_max, lat_max)
    date_start: str,    # YYYY-MM-DD
    date_end:   str,    # YYYY-MM-DD
    cloud_max:  int = 30,
    n_scenes:   int = 5,
) -> list[dict]:
    """Search for Sentinel-2 L2A scenes via Copernicus Data Space OData API.

    Args:
        username:   CDSE username (register free at https://dataspace.copernicus.eu/)
        password:   CDSE password.
        bbox:       Bounding box (lon_min, lat_min, lon_max, lat_max) in WGS84.
        date_start: Start date string 'YYYY-MM-DD'.
        date_end:   End date string 'YYYY-MM-DD'.
        cloud_max:  Maximum cloud cover percentage to include.
        n_scenes:   Maximum number of scenes to return.

    Returns:
        List of product metadata dicts with 'id', 'name', 'cloud_cover'.
    """
    import requests

    lon_min, lat_min, lon_max, lat_max = bbox
    poly = (
        f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
        f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))"
    )

    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-2' and "
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.DoubleAttribute/Value le {cloud_max}) and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{poly}') and "
            f"ContentDate/Start gt {date_start}T00:00:00.000Z and "
            f"ContentDate/Start lt {date_end}T23:59:59.000Z and "
            f"contains(Name,'MSIL2A')"
        ),
        "$orderby": "ContentDate/Start desc",
        "$top": n_scenes,
        "$expand": "Attributes",
    }

    logger.info("Searching CDSE for Sentinel-2 L2A scenes …")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("value", [])

    results = []
    for item in items:
        cloud = next(
            (a["Value"] for a in item.get("Attributes", [])
             if a.get("Name") == "cloudCover"),
            None,
        )
        results.append({"id": item["Id"], "name": item["Name"], "cloud_cover": cloud})
        logger.info("  Found: %s  cloud=%.1f%%", item["Name"], cloud or 0)

    return results


def download_sentinel2_product(
    product_id: str,
    product_name: str,
    username: str,
    password: str,
    out_dir: Path,
) -> Path:
    """Download a single Sentinel-2 product (SAFE archive) from CDSE.

    Args:
        product_id:   CDSE product UUID.
        product_name: Human-readable product name (used for the filename).
        username:     CDSE username.
        password:     CDSE password.
        out_dir:      Destination directory.

    Returns:
        Path to the downloaded .zip / .SAFE file.
    """
    import requests
    from tqdm import tqdm

    # Step 1: Get access token
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    token_resp = requests.post(
        token_url,
        data={
            "grant_type": "password",
            "client_id":  "cdse-public",
            "username":   username,
            "password":   password,
        },
        timeout=30,
    )
    token_resp.raise_for_status()
    token = token_resp.json()["access_token"]

    # Step 2: Download product
    dl_url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{product_name}.zip"

    logger.info("Downloading %s …", product_name)
    with requests.get(dl_url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    return out_path


def extract_bands_from_safe(safe_dir: Path, out_path: Path) -> None:
    """Extract B02, B03, B04, B08 from a Sentinel-2 .SAFE archive and merge them.

    Sentinel-2 .SAFE structure:
        <SAFE>/GRANULE/<granule_id>/IMG_DATA/R10m/
            *_B02_10m.jp2   (Blue)
            *_B03_10m.jp2   (Green)
            *_B04_10m.jp2   (Red)
            *_B08_10m.jp2   (NIR)

    Args:
        safe_dir:  Path to extracted .SAFE directory.
        out_path:  Destination 4-band GeoTIFF.
    """
    import rasterio

    band_patterns = {
        "blue":  "*_B02_10m.jp2",
        "green": "*_B03_10m.jp2",
        "red":   "*_B04_10m.jp2",
        "nir":   "*_B08_10m.jp2",
    }
    band_order = ["red", "green", "blue", "nir"]

    found: dict[str, Path] = {}
    for band_name, pattern in band_patterns.items():
        matches = list(safe_dir.rglob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"Band file '{pattern}' not found under {safe_dir}"
            )
        found[band_name] = matches[0]

    bands: list[np.ndarray] = []
    profile = None
    import numpy as np
    for band_name in band_order:
        with rasterio.open(found[band_name]) as src:
            bands.append(src.read(1))
            if profile is None:
                profile = src.profile.copy()

    profile.update({
        "count":  4,
        "dtype":  "uint16",
        "driver": "GTiff",
        "compress": "lzw",
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, b in enumerate(bands, 1):
            dst.write(b, i)

    logger.info("4-band GeoTIFF saved → %s", out_path)


def download_and_convert(
    username: str,
    password: str,
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    cloud_max: int = 30,
    n_scenes: int = 5,
) -> int:
    """Full pipeline: search → download → extract → 4-band GeoTIFF.

    Returns:
        Number of scenes successfully downloaded and converted.
    """
    import zipfile, shutil

    scenes   = search_sentinel2_scenes(
        username, password, bbox, date_start, date_end, cloud_max, n_scenes
    )
    if not scenes:
        logger.warning("No scenes found matching your criteria.")
        return 0

    tmp_dir = PROJECT_ROOT / "data" / "_sentinel2_tmp"
    count = 0

    for scene in scenes:
        try:
            zip_path = download_sentinel2_product(
                scene["id"], scene["name"], username, password, tmp_dir
            )
            safe_dir = tmp_dir / scene["name"]
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(safe_dir)

            safe_path = next(safe_dir.rglob("*.SAFE"), None)
            if safe_path is None:
                safe_path = safe_dir

            out_tif = RAW_DIR / f"s2_{scene['name'][:20]}.tif"
            extract_bands_from_safe(safe_path, out_tif)

            logger.info(
                "✅ Scene %d: %s → %s  (cloud=%.1f%%)",
                count + 1, scene["name"][:20], out_tif.name, scene["cloud_cover"] or 0,
            )
            count += 1

        except Exception as exc:
            logger.error("Failed to process scene '%s': %s", scene["name"], exc)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("Downloaded %d / %d scenes → data/raw/", count, len(scenes))
    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 L2A imagery from CDSE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--username",        required=True,  help="CDSE username")
    parser.add_argument("--password",        required=True,  help="CDSE password")
    parser.add_argument("--bbox",            required=True,  help="lon_min,lat_min,lon_max,lat_max")
    parser.add_argument("--date_start",      required=True,  help="YYYY-MM-DD")
    parser.add_argument("--date_end",        required=True,  help="YYYY-MM-DD")
    parser.add_argument("--cloud_cover_max", type=int, default=30)
    parser.add_argument("--n_scenes",        type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    _check_dependencies()
    args = _parse_args()
    bbox_vals = tuple(float(x) for x in args.bbox.split(","))
    if len(bbox_vals) != 4:
        logger.error("--bbox must have exactly 4 comma-separated values")
        sys.exit(1)
    download_and_convert(
        username   = args.username,
        password   = args.password,
        bbox       = bbox_vals,
        date_start = args.date_start,
        date_end   = args.date_end,
        cloud_max  = args.cloud_cover_max,
        n_scenes   = args.n_scenes,
    )
