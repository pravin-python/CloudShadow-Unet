"""
Module 6 — Streamlit Interactive Dashboard
==========================================
Serves the CloudShadow-UNet inference pipeline as a web app.

Features
--------
- Drag-and-drop upload of 4-band GeoTIFF files
- Cached model loading (@st.cache_resource) — model loaded once per session
- Cached raster reading (@st.cache_data) — prevents redundant disk I/O
- Side-by-side RGB preview vs predicted mask (folium/leafmap or static render)
- Interactive image-comparison slider via streamlit-image-comparison
- Real-time area statistics (km² per class, cloud/shadow fractions)
- Download button for the predicted GeoTIFF mask

Architecture notes
------------------
Streamlit reruns the entire script on each widget interaction.  The two
caching decorators are critical for performance:

    @st.cache_resource — for objects that cannot be serialised (Keras model,
                         rasterio file handles).  Never recreated mid-session.

    @st.cache_data     — for pure data arrays (numpy, pandas DataFrames).
                         Recomputed only when input arguments change.

Run:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path to resolve 'src' module
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)

# ─── page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="CloudShadow-UNet",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── constants ────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    0: np.array([128, 128, 128], dtype=np.uint8),   # Background — grey
    1: np.array([255, 255, 255], dtype=np.uint8),   # Cloud      — white
    2: np.array([30,  60,  120], dtype=np.uint8),   # Shadow     — dark blue
}
CLASS_NAMES = {0: "Background", 1: "Cloud", 2: "Cloud Shadow"}

DEFAULT_MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", "models/best_weights.h5")
)
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", 256))
OVERLAP = float(os.environ.get("OVERLAP", 0.25))
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", 3))


# ─── cached resource: model ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading segmentation model …")
def load_model(model_path: str):
    """Load and cache the Keras U-Net model.

    @st.cache_resource ensures the model is loaded exactly once per
    Streamlit server process, regardless of how many times the script
    reruns due to widget interactions.

    Args:
        model_path: Path to the saved Keras model (h5 or SavedModel dir).

    Returns:
        Loaded Keras model, or None if the file does not exist.
    """
    import tensorflow as tf
    from src.model.losses import CombinedLoss, DiceCoefficient, MeanIoU

    path = Path(model_path)
    if not path.exists():
        return None

    model = tf.keras.models.load_model(
        str(path),
        custom_objects={
            "CombinedLoss": CombinedLoss,
            "DiceCoefficient": DiceCoefficient,
            "MeanIoU": MeanIoU,
        },
    )
    return model


# ─── cached data: raster reading ──────────────────────────────────────────────

@st.cache_data(show_spinner="Reading GeoTIFF …")
def read_uploaded_geotiff(file_bytes: bytes) -> tuple[np.ndarray, dict]:
    """Read a 4-band GeoTIFF from in-memory bytes and return normalised array.

    @st.cache_data caches based on the content of file_bytes.  Re-uploading
    the same file triggers no redundant I/O.  A new file invalidates the cache
    automatically.

    Args:
        file_bytes: Raw bytes of the uploaded GeoTIFF.

    Returns:
        image:   float32 (H, W, 4) array, range [0, 1].
        profile: rasterio profile dict (CRS, transform, …).
    """
    import rasterio
    from src.preprocessing.preprocess import REFLECTANCE_SCALE, apply_clahe_per_band

    with rasterio.MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            if src.count < 4:
                st.error(
                    f"Uploaded file has {src.count} band(s). "
                    "CloudShadow-UNet requires exactly 4 bands (R, G, B, NIR)."
                )
                st.stop()
            raw = src.read([1, 2, 3, 4], out_dtype=np.float32)
            profile = src.profile.copy()

    image = np.transpose(raw, (1, 2, 0))
    image = np.clip(image / REFLECTANCE_SCALE, 0.0, 1.0)
    image = apply_clahe_per_band(image)
    return image, profile


# ─── rendering helpers ────────────────────────────────────────────────────────

def class_map_to_rgb(class_map: np.ndarray) -> np.ndarray:
    """Convert a (H, W) integer label map to a (H, W, 3) uint8 RGB image.

    Args:
        class_map: int32 (H, W) array with values in {0, 1, 2}.

    Returns:
        uint8 (H, W, 3) colour-coded mask.
    """
    rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_id, colour in CLASS_COLORS.items():
        rgb[class_map == cls_id] = colour
    return rgb


def image_to_rgb_preview(image: np.ndarray) -> np.ndarray:
    """Extract the RGB bands (first 3) from a 4-band image for display.

    Args:
        image: float32 (H, W, 4) normalised RGBNIR array.

    Returns:
        uint8 (H, W, 3) gamma-corrected display image.
    """
    rgb = image[:, :, :3]
    # Gamma correction (γ=0.5) brightens low-reflectance dark imagery
    # for perceptual display — does not affect model input.
    rgb_gamma = np.power(np.clip(rgb, 0.0, 1.0), 0.5)
    return (rgb_gamma * 255).astype(np.uint8)


@st.cache_data(show_spinner="Running segmentation …")
def run_inference_cached(
    image: np.ndarray,
    model_path: str,
) -> np.ndarray:
    """Run sliding-window inference and return the predicted class map.

    Cached so repeated renders (e.g., switching tabs) do not re-run inference.

    Args:
        image:      float32 (H, W, 4) normalised image.
        model_path: Path to the model (used as part of the cache key).

    Returns:
        int32 (H, W) predicted class label map.
    """
    from src.inference.predict import sliding_window_predict

    model = load_model(model_path)
    if model is None:
        st.error(f"Model not found at {model_path}.")
        st.stop()

    class_map = sliding_window_predict(
        model, image,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
    )
    return class_map


def mask_to_geotiff_bytes(
    class_map: np.ndarray,
    profile: dict,
) -> bytes:
    """Serialise a predicted class map to GeoTIFF bytes for download.

    Args:
        class_map: int32 (H, W) predicted label array.
        profile:   rasterio profile from the source GeoTIFF.

    Returns:
        GeoTIFF bytes that preserve source CRS and affine transform.
    """
    import rasterio

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", compress="lzw", nodata=255, driver="GTiff")

    buf = io.BytesIO()
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**out_profile) as dst:
            dst.write(class_map.astype(np.uint8), 1)
        buf.write(memfile.read())

    return buf.getvalue()


# ─── sidebar ──────────────────────────────────────────────────────────────────

def _render_sidebar() -> str:
    """Render sidebar controls and return the model path string."""
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
        use_container_width=True,
    )
    st.sidebar.title("CloudShadow-UNet")
    st.sidebar.caption("Satellite cloud & shadow segmentation")
    st.sidebar.markdown("---")

    model_path = st.sidebar.text_input(
        "Model path",
        value=str(DEFAULT_MODEL_PATH),
        help="Path to trained Keras model (.h5 or SavedModel directory)",
    )

    st.sidebar.markdown("### Inference settings")
    patch_size_display = st.sidebar.selectbox(
        "Patch size", options=[128, 256, 384, 512], index=1
    )
    overlap_display = st.sidebar.slider(
        "Overlap fraction", min_value=0.0, max_value=0.5, value=0.25, step=0.05
    )

    # Store in session state so inference function can read them
    st.session_state["patch_size"] = patch_size_display
    st.session_state["overlap"] = overlap_display

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Class legend**\n\n"
        "🔲 Grey — Background\n\n"
        "⬜ White — Cloud\n\n"
        "🟦 Dark blue — Cloud Shadow"
    )

    return model_path


# ─── statistics panel ─────────────────────────────────────────────────────────

def _render_statistics(class_map: np.ndarray, profile: dict) -> None:
    """Render per-class area statistics derived from the prediction array."""
    from src.inference.predict import compute_area_statistics

    stats = compute_area_statistics(class_map, profile)

    if not stats:
        st.warning(
            "Could not compute km² statistics — spatial metadata missing from "
            "the uploaded file.  Showing pixel-count statistics instead."
        )
        for cls_id, name in CLASS_NAMES.items():
            px = int(np.sum(class_map == cls_id))
            st.metric(name, f"{px:,} px")
        return

    st.subheader("Geospatial Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total scene area", f"{stats.get('total_scene_km2', 0):,.1f} km²")
    col2.metric(
        "Cloud cover",
        f"{stats.get('cloud_km2', 0):,.2f} km²",
        delta=f"{stats.get('cloud_fraction', 0) * 100:.1f} % of scene",
    )
    col3.metric(
        "Cloud shadow",
        f"{stats.get('shadow_km2', 0):,.2f} km²",
        delta=f"{stats.get('shadow_fraction', 0) * 100:.1f} % of scene",
    )
    col4.metric("Background", f"{stats.get('background_km2', 0):,.1f} km²")

    # Class distribution bar chart
    import pandas as pd

    df = pd.DataFrame(
        {
            "Class": [CLASS_NAMES[c] for c in range(NUM_CLASSES)],
            "Area (km²)": [
                stats.get("background_km2", 0),
                stats.get("cloud_km2", 0),
                stats.get("shadow_km2", 0),
            ],
        }
    )
    st.bar_chart(df.set_index("Class"))


# ─── map visualisation ────────────────────────────────────────────────────────

def _render_leafmap(
    image: np.ndarray,
    class_map: np.ndarray,
    profile: dict,
) -> None:
    """Render the RGB image and predicted mask on an interactive leafmap.

    If leafmap is unavailable (environment without browser dependencies),
    falls back to a side-by-side static image comparison.

    Args:
        image:     float32 (H, W, 4) normalised RGBNIR image.
        class_map: int32   (H, W)    predicted class label map.
        profile:   rasterio profile with CRS and transform.
    """
    try:
        import leafmap
        import tempfile, rasterio
        from rasterio.warp import transform_bounds
        from rasterio.crs import CRS

        # Write RGB preview GeoTIFF
        rgb = image_to_rgb_preview(image)
        wgs84 = CRS.from_epsg(4326)
        src_crs = profile.get("crs", wgs84)
        transform = profile.get("transform")

        with tempfile.NamedTemporaryFile(suffix="_rgb.tif", delete=False) as f:
            rgb_path = f.name
        with tempfile.NamedTemporaryFile(suffix="_mask.tif", delete=False) as f:
            mask_path = f.name

        rgb_profile = profile.copy()
        rgb_profile.update(count=3, dtype="uint8", driver="GTiff")
        with rasterio.open(rgb_path, "w", **rgb_profile) as dst:
            dst.write(np.transpose(rgb, (2, 0, 1)))

        mask_rgb = class_map_to_rgb(class_map)
        mask_profile = profile.copy()
        mask_profile.update(count=3, dtype="uint8", driver="GTiff")
        with rasterio.open(mask_path, "w", **mask_profile) as dst:
            dst.write(np.transpose(mask_rgb, (2, 0, 1)))

        m = leafmap.Map(center=[0, 0], zoom=2)
        m.add_raster(rgb_path, layer_name="Satellite Image", opacity=1.0)
        m.add_raster(mask_path, layer_name="Predicted Mask", opacity=0.6)
        m.add_layer_manager()
        m.to_streamlit(height=550)

    except Exception as exc:
        logger.warning("leafmap render failed (%s); falling back to static view.", exc)
        col_img, col_mask = st.columns(2)
        with col_img:
            st.image(
                image_to_rgb_preview(image),
                caption="RGB preview (Bands 1-3)",
                use_container_width=True,
            )
        with col_mask:
            st.image(
                class_map_to_rgb(class_map),
                caption="Predicted mask",
                use_container_width=True,
            )


# ─── image comparison slider ──────────────────────────────────────────────────

def _render_comparison_slider(
    image: np.ndarray,
    class_map: np.ndarray,
) -> None:
    """Render an interactive before/after image comparison slider.

    Requires the streamlit-image-comparison package.
    Falls back to st.columns display if unavailable.
    """
    try:
        from streamlit_image_comparison import image_comparison  # type: ignore

        image_comparison(
            img1=image_to_rgb_preview(image),
            img2=class_map_to_rgb(class_map),
            label1="Original satellite image",
            label2="Predicted cloud/shadow mask",
            width=900,
        )
    except ImportError:
        st.info(
            "Install `streamlit-image-comparison` for an interactive slider: "
            "`pip install streamlit-image-comparison`"
        )
        _render_leafmap.__wrapped__ if False else None
        col1, col2 = st.columns(2)
        col1.image(image_to_rgb_preview(image), caption="Original", use_container_width=True)
        col2.image(class_map_to_rgb(class_map), caption="Predicted Mask", use_container_width=True)


# ─── main app ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the Streamlit dashboard."""
    model_path = _render_sidebar()

    st.title("CloudShadow-UNet — Cloud & Shadow Segmentation")
    st.markdown(
        "Upload a **4-band GeoTIFF** (R, G, B, NIR) to segment clouds and "
        "shadows using the U-Net deep learning model.  The output mask is "
        "georeferenced and downloadable for use in QGIS or ArcGIS."
    )

    # ── model status indicator ────────────────────────────────────────────────
    model = load_model(model_path)
    if model is None:
        st.warning(
            f"No trained model found at `{model_path}`.  "
            "Train the model first with:\n\n"
            "```bash\npython src/training/train.py --config configs/unet_baseline.yaml\n```"
        )
    else:
        st.success(f"Model loaded — {model.count_params():,} parameters")

    st.markdown("---")

    # ── file upload ───────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a 4-band GeoTIFF",
        type=["tif", "tiff"],
        help="Sentinel-2 or Landsat 8 Level-2A GeoTIFF with R, G, B, NIR bands",
    )

    if uploaded is None:
        st.info("Upload a GeoTIFF to begin segmentation.")
        return

    file_bytes = uploaded.read()
    image, profile = read_uploaded_geotiff(file_bytes)

    h, w = image.shape[:2]
    st.caption(
        f"Scene: **{uploaded.name}**  |  Size: **{w} × {h}** px  |  "
        f"CRS: `{profile.get('crs', 'unknown')}`"
    )

    # ── inference ─────────────────────────────────────────────────────────────
    if model is None:
        st.error("Cannot run inference without a trained model.")
        return

    with st.spinner("Running sliding-window inference …"):
        class_map = run_inference_cached(image, model_path)

    st.success("Inference complete!")
    st.markdown("---")

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab_map, tab_compare, tab_stats, tab_download = st.tabs(
        ["Interactive Map", "Image Comparison", "Statistics", "Download"]
    )

    with tab_map:
        st.subheader("Interactive Map View")
        _render_leafmap(image, class_map, profile)

    with tab_compare:
        st.subheader("Before / After Comparison")
        _render_comparison_slider(image, class_map)

    with tab_stats:
        _render_statistics(class_map, profile)

    with tab_download:
        st.subheader("Download Predicted Mask")
        st.markdown(
            "The downloaded GeoTIFF embeds the original CRS and affine "
            "transform so it aligns perfectly with your source imagery in QGIS."
        )

        mask_bytes = mask_to_geotiff_bytes(class_map, profile)
        st.download_button(
            label="Download predicted mask (.tif)",
            data=mask_bytes,
            file_name=f"{Path(uploaded.name).stem}_cloud_mask.tif",
            mime="image/tiff",
        )

        st.markdown("**Label encoding:**")
        st.table(
            {
                "Value": [0, 1, 2],
                "Class": ["Background", "Cloud", "Cloud Shadow"],
                "Colour": ["Grey (128,128,128)", "White (255,255,255)", "Dark Blue (30,60,120)"],
            }
        )


if __name__ == "__main__":
    main()
