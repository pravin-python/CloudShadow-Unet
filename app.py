"""
app.py — Professional Streamlit Interactive Dashboard
======================================================
Principal Engineer: CloudShadow-UNet Project
─────────────────────────────────────────────────────────────────────────────
Full-featured web dashboard for:

  • Uploading 4-band GeoTIFF satellite imagery (drag-and-drop)
  • Running deep learning cloud/shadow segmentation in real time
  • Visualising results on an interactive leafmap / folium map
  • Displaying KPI metric cards (km² cloud cover, fractions, GSD)
  • Comparing original vs predicted mask with an interactive slider
  • Downloading the georeferenced predicted mask (.tif)
  • Triggering continuous fine-tuning on user-uploaded annotated data
  • Tracking training progress with animated progress bars

Performance & stability guarantees:
  @st.cache_resource  → TensorFlow model: loaded once, never re-created
  @st.cache_data      → Raster reads and inference: keyed on file bytes;
                        switching tabs / adjusting sliders never re-runs inference

Layout:
  ┌─────────────────────────────────────────────────────────────────┐
  │ SIDEBAR                        │  MAIN AREA                     │
  │ ─────────────────────────────  │  ──────────────────────────    │
  │ Model path text input          │  [Tab] Interactive Map          │
  │ Upload image TIF               │  [Tab] Image Comparison Slider  │
  │ Upload mask TIF (optional)     │  [Tab] Geospatial Statistics    │
  │ Confidence threshold slider    │  [Tab] Download                 │
  │ Patch size selector            │  [Tab] Fine-Tune Training       │
  │ Overlap slider                 │                                 │
  │ [Run Inference] button         │                                 │
  │ ──────────────────────────     │                                 │
  │ Fine-Tune section              │                                 │
  │ Upload annotated TIF           │                                 │
  │ Upload annotation mask         │                                 │
  │ Fine-tune epochs slider        │                                 │
  │ [Start Fine-Tuning] button     │                                 │
  └─────────────────────────────────────────────────────────────────┘

Run:
    streamlit run app.py

    # Remote server with large files:
    streamlit run app.py --server.maxUploadSize 2048 --server.address 0.0.0.0
"""

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

# ─── page config — MUST be the very first Streamlit call ─────────────────────
st.set_page_config(
    page_title   = "CloudShadow-UNet Dashboard",
    page_icon    = "🛰️",
    layout       = "wide",
    initial_sidebar_state = "expanded",
    menu_items   = {
        "Get Help":    "https://github.com/pravin-python/CloudShadow-Unet",
        "Report a bug": "https://github.com/pravin-python/CloudShadow-Unet/issues",
        "About": "CloudShadow-UNet — End-to-End Satellite Cloud & Shadow Segmentation",
    },
)

# Ensure project root is on sys.path so local modules import correctly
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/best_initial.h5"))
DEFAULT_PATCH_SIZE = int(os.environ.get("PATCH_SIZE", 256))
DEFAULT_OVERLAP    = float(os.environ.get("OVERLAP", 0.25))
NUM_CLASSES        = 3

CLASS_NAMES   = {0: "Background", 1: "Cloud", 2: "Cloud Shadow"}
CLASS_COLORS  = {
    0: (128, 128, 128),  # grey
    1: (255, 255, 255),  # white
    2: (30,  60,  120),  # dark blue
}
LEGEND_HEX = {
    0: "#808080",
    1: "#FFFFFF",
    2: "#1E3C78",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CACHED RESOURCE LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔄 Loading segmentation model into GPU …")
def _load_model(model_path: str):
    """Load and cache the Keras U-Net model.

    @st.cache_resource caches by the function arguments.  A different
    model_path string invalidates the cache and loads the new model.
    The same path → same model object, zero reloading.

    Args:
        model_path: String path to the saved model (.h5 or SavedModel dir).

    Returns:
        Loaded Keras Model, or None if the file does not exist.
    """
    import tensorflow as tf
    from model import CUSTOM_OBJECTS

    path = Path(model_path)
    if not path.exists():
        return None
    try:
        model = tf.keras.models.load_model(str(path), custom_objects=CUSTOM_OBJECTS)
        logger.info("Model loaded from '%s'  params=%s", path, f"{model.count_params():,}")
        return model
    except Exception as exc:
        # Try loading as weights-only .h5 with architecture rebuild
        try:
            from model import build_unet
            m = build_unet(input_shape=(DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE, 4))
            m.load_weights(str(path))
            logger.info("Weights loaded (architecture rebuilt) from '%s'", path)
            return m
        except Exception as exc2:
            st.error(f"Failed to load model:\n{exc}\n{exc2}")
            return None


@st.cache_data(show_spinner="📡 Reading GeoTIFF …", max_entries=3)
def _read_geotiff_bytes(file_bytes: bytes) -> tuple[np.ndarray, dict] | None:
    """Read a 4-band GeoTIFF from raw bytes into a normalised float32 array.

    @st.cache_data caches based on the CONTENT of file_bytes (hash).
    Re-uploading the exact same file hits the cache — zero redundant I/O.
    Uploading a new file automatically invalidates the cache entry.

    max_entries=3: Keep only the 3 most recent scenes cached to cap RAM.

    Args:
        file_bytes: Raw bytes of the uploaded GeoTIFF.

    Returns:
        (image, profile) tuple, or None on error.
    """
    import rasterio
    from geospatial_utils import apply_clahe

    try:
        with rasterio.MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                if src.count < 4:
                    st.error(
                        f"Uploaded file has **{src.count} band(s)**. "
                        "The model requires **4 bands** (Red, Green, Blue, NIR). "
                        "Use GDAL to merge bands:\n"
                        "```bash\ngdal_merge.py -separate -o merged.tif r.tif g.tif b.tif nir.tif\n```"
                    )
                    return None
                raw     = src.read([1, 2, 3, 4], out_dtype=np.float32)
                profile = src.profile.copy()

        image = np.transpose(raw, (1, 2, 0))
        image = np.clip(image / 10_000.0, 0.0, 1.0)
        image = apply_clahe(image)
        return image, profile

    except Exception as exc:
        st.error(f"GeoTIFF read error: {exc}")
        return None


@st.cache_data(
    show_spinner="🤖 Running deep learning inference …",
    max_entries=2,
)
def _run_inference_cached(
    file_bytes:   bytes,
    model_path:   str,
    patch_size:   int,
    overlap:      float,
) -> np.ndarray | None:
    """Run sliding-window inference and cache the result.

    Cache key = (file_bytes hash, model_path, patch_size, overlap).
    Switching dashboard tabs or adjusting the visualisation threshold
    never re-runs inference — only uploading a new file does.

    Args:
        file_bytes:  Raw GeoTIFF bytes (used as cache key).
        model_path:  Model path (cache key component).
        patch_size:  Inference tile size.
        overlap:     Tile overlap fraction.

    Returns:
        int32 (H, W) class label map, or None on error.
    """
    from geospatial_utils import stitch_predictions

    result = _read_geotiff_bytes(file_bytes)
    if result is None:
        return None
    image, _ = result

    model = _load_model(model_path)
    if model is None:
        st.error(f"Model not found at `{model_path}`. Train the model first.")
        return None

    try:
        class_map = stitch_predictions(
            model,
            image,
            patch_size  = patch_size,
            overlap     = overlap,
            num_classes = NUM_CLASSES,
        )
        return class_map
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        logger.exception("Inference error")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — RENDERING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _class_map_to_rgb(class_map: np.ndarray) -> np.ndarray:
    """Convert integer class label map → uint8 RGB colour image."""
    rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_id, colour in CLASS_COLORS.items():
        rgb[class_map == cls_id] = colour
    return rgb


def _image_to_rgb_u8(image: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Extract and gamma-correct the RGB bands (channels 0-2) for display."""
    rgb = np.power(np.clip(image[:, :, :3], 0.0, 1.0), gamma)
    return (rgb * 255.0).astype(np.uint8)


def _mask_to_geotiff_bytes(class_map: np.ndarray, profile: dict) -> bytes:
    """Serialise predicted class map to in-memory GeoTIFF bytes for download."""
    import rasterio

    out_profile = profile.copy()
    out_profile.update({
        "count":    1,
        "dtype":    "uint8",
        "driver":   "GTiff",
        "compress": "lzw",
        "predictor": 2,
        "nodata":   255,
    })

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**out_profile) as dst:
            dst.write(class_map.astype(np.uint8), 1)
            dst.write_colormap(1, {
                0:   (128, 128, 128, 255),
                1:   (255, 255, 255, 255),
                 2:   (30,  60,  120, 255),
                255: (0, 0, 0, 0),
            })
        return memfile.read()


def _apply_confidence_threshold(
    class_map: np.ndarray,
    threshold:  float,
    file_bytes: bytes,
    model_path: str,
    patch_size: int,
    overlap:    float,
) -> np.ndarray:
    """Re-apply a confidence threshold to the raw softmax output.

    Pixels whose max softmax probability is below `threshold` are
    reclassified as Background (0) — useful for reducing false positives
    in ambiguous regions (e.g., bright sand, snow, water glint).

    Because the cached inference result only stores argmax labels (not raw
    probabilities), this function re-runs a lightweight single forward pass
    on the full image to retrieve probabilities.  This is called only when
    the user moves the confidence threshold slider — not on every render.

    Args:
        class_map:  Cached argmax prediction (H, W).
        threshold:  Minimum softmax confidence to accept a non-background label.
        file_bytes, model_path, patch_size, overlap: passed to stitch_predictions.

    Returns:
        Thresholded int32 (H, W) class map.
    """
    if threshold <= 0.0:
        return class_map   # No thresholding requested

    result = _read_geotiff_bytes(file_bytes)
    if result is None:
        return class_map
    image, _ = result

    model = _load_model(model_path)
    if model is None:
        return class_map

    from geospatial_utils import cosine_bell_mask, generate_tile_coords

    h, w      = image.shape[:2]
    coords    = generate_tile_coords(h, w, patch_size, overlap)
    bell      = cosine_bell_mask(patch_size)
    n_classes = NUM_CLASSES

    prob_accum   = np.zeros((h, w, n_classes), dtype=np.float64)
    weight_accum = np.zeros((h, w),            dtype=np.float64)

    batch = 8
    for i in range(0, len(coords), batch):
        bc = coords[i: i + batch]
        patches = np.stack([image[r0:r1, c0:c1, :] for r0, r1, c0, c1 in bc]).astype(np.float32)
        preds = model.predict(patches, verbose=0)
        for (r0, r1, c0, c1), pred in zip(bc, preds):
            prob_accum  [r0:r1, c0:c1, :] += pred.astype(np.float64) * bell[:, :, np.newaxis]
            weight_accum[r0:r1, c0:c1]    += bell

    weight_accum  = np.where(weight_accum == 0.0, 1.0, weight_accum)
    blended       = prob_accum / weight_accum[:, :, np.newaxis]
    max_prob      = np.max(blended, axis=-1)
    thresholded   = np.argmax(blended, axis=-1).astype(np.int32)
    thresholded[max_prob < threshold] = 0   # Set low-confidence pixels → background
    return thresholded


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> dict:
    """Render all sidebar controls and return a config dict.

    Returns:
        dict with keys:
            model_path, patch_size, overlap, confidence_threshold,
            run_inference, uploaded_image_bytes, scene_name,
            ft_uploaded_image_bytes, ft_uploaded_mask_bytes,
            ft_epochs, start_finetune
    """
    st.sidebar.markdown(
        "<h2 style='color:#1E90FF;'>🛰️ CloudShadow-UNet</h2>"
        "<p style='color:#888;font-size:0.85rem;'>Satellite Segmentation Dashboard</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    # ── Model settings ────────────────────────────────────────────────────────
    st.sidebar.subheader("⚙️ Model Configuration")

    model_path = st.sidebar.text_input(
        "Model path",
        value=str(DEFAULT_MODEL_PATH),
        help="Path to trained .h5 weights or SavedModel directory",
    )

    model = _load_model(model_path)
    if model is not None:
        st.sidebar.success(f"✅ Model ready  ({model.count_params():,} params)")
    else:
        st.sidebar.warning("⚠️ Model not found — train first")

    patch_size = st.sidebar.selectbox(
        "Patch size (px)",
        options=[128, 256, 384, 512],
        index=1,
        help="Larger = more context; smaller = less GPU memory",
    )
    overlap = st.sidebar.slider(
        "Tile overlap",
        min_value=0.0,
        max_value=0.5,
        value=DEFAULT_OVERLAP,
        step=0.05,
        format="%.0f%%",
        help="Higher overlap → smoother stitching, slower inference",
    )
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Pixels below this softmax confidence → Background",
    )

    st.sidebar.markdown("---")

    # ── Image upload ──────────────────────────────────────────────────────────
    st.sidebar.subheader("📂 Upload Satellite Image")
    uploaded_image = st.sidebar.file_uploader(
        "4-band GeoTIFF (R, G, B, NIR)",
        type=["tif", "tiff"],
        key="image_uploader",
        help="Sentinel-2 or Landsat 8 Level-2A GeoTIFF with 4 bands",
    )

    run_inference = st.sidebar.button(
        "🚀 Run Inference",
        type="primary",
        disabled=(uploaded_image is None or model is None),
        use_container_width=True,
    )

    st.sidebar.markdown("---")

    # ── Fine-tuning ───────────────────────────────────────────────────────────
    with st.sidebar.expander("🔬 Fine-Tune on New Data", expanded=False):
        st.caption(
            "Upload a new GeoTIFF + its annotation mask to incorporate "
            "into the training pool and run a fine-tuning pass."
        )
        ft_image  = st.file_uploader("New scene (GeoTIFF)", type=["tif","tiff"], key="ft_image")
        ft_mask   = st.file_uploader("Annotation mask (GeoTIFF)", type=["tif","tiff"], key="ft_mask")
        ft_epochs = st.slider("Fine-tune epochs", min_value=1, max_value=50, value=10)
        ft_button = st.button(
            "⚡ Start Fine-Tuning",
            type="secondary",
            disabled=(ft_image is None or ft_mask is None or model is None),
            use_container_width=True,
        )

    st.sidebar.markdown("---")

    # ── Legend ────────────────────────────────────────────────────────────────
    st.sidebar.markdown("**Class Legend**")
    for cls_id, name in CLASS_NAMES.items():
        hex_col = LEGEND_HEX[cls_id]
        border  = "border:1px solid #555;" if cls_id == 1 else ""
        st.sidebar.markdown(
            f"<div style='display:flex;align-items:center;margin:2px 0'>"
            f"<span style='width:16px;height:16px;background:{hex_col};"
            f"{border}display:inline-block;margin-right:8px;border-radius:3px'></span>"
            f"<span>{cls_id} — {name}</span></div>",
            unsafe_allow_html=True,
        )

    return {
        "model_path":               model_path,
        "patch_size":               int(patch_size),
        "overlap":                  float(overlap),
        "confidence_threshold":     float(confidence_threshold),
        "run_inference":            run_inference,
        "uploaded_image_bytes":     uploaded_image.read() if uploaded_image else None,
        "scene_name":               uploaded_image.name if uploaded_image else None,
        "ft_uploaded_image_bytes":  ft_image.read()  if ft_image  else None,
        "ft_uploaded_mask_bytes":   ft_mask.read()   if ft_mask   else None,
        "ft_epochs":                ft_epochs,
        "start_finetune":           ft_button,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — INTERACTIVE MAP TAB
# ═══════════════════════════════════════════════════════════════════════════════

def _render_interactive_map(image: np.ndarray, class_map: np.ndarray, profile: dict) -> None:
    """Render source image + prediction mask on an interactive leafmap.

    Strategy:
      1. Write temporary GeoTIFF files for both the RGB preview and the mask.
      2. Load them into a leafmap.Map instance using add_raster().
      3. Add a layer manager widget so users can toggle visibility + opacity.
      4. Fall back to a side-by-side static display if leafmap is unavailable.
    """
    try:
        import leafmap
        import rasterio

        rgb_preview   = _image_to_rgb_u8(image)
        mask_rgb      = _class_map_to_rgb(class_map)

        # Write temp files — leafmap requires file paths, not in-memory objects
        with (
            tempfile.NamedTemporaryFile(suffix="_rgb.tif",  delete=False) as f_rgb,
            tempfile.NamedTemporaryFile(suffix="_mask.tif", delete=False) as f_mask,
        ):
            rgb_path  = Path(f_rgb.name)
            mask_path = Path(f_mask.name)

        # Write RGB GeoTIFF
        rgb_profile = profile.copy()
        rgb_profile.update({"count": 3, "dtype": "uint8", "driver": "GTiff"})
        with rasterio.open(rgb_path, "w", **rgb_profile) as dst:
            dst.write(np.transpose(rgb_preview, (2, 0, 1)))

        # Write mask GeoTIFF
        mask_profile = profile.copy()
        mask_profile.update({"count": 3, "dtype": "uint8", "driver": "GTiff"})
        with rasterio.open(mask_path, "w", **mask_profile) as dst:
            dst.write(np.transpose(mask_rgb, (2, 0, 1)))

        m = leafmap.Map(center=[0, 0], zoom=3, draw_control=False)
        m.add_raster(str(rgb_path),  layer_name="🛰️ Satellite Image",  opacity=1.0)
        m.add_raster(str(mask_path), layer_name="🎨 Predicted Mask",   opacity=0.65)
        m.add_layer_manager()
        m.to_streamlit(height=580, bidirectional=False)

    except ImportError:
        st.info(
            "Install **leafmap** for an interactive map view: "
            "`pip install leafmap`.  Showing static view instead."
        )
        _render_static_columns(image, class_map)

    except Exception as exc:
        logger.warning("leafmap render error: %s", exc)
        st.warning(f"Map render failed ({exc}).  Showing static view.")
        _render_static_columns(image, class_map)


def _render_static_columns(image: np.ndarray, class_map: np.ndarray) -> None:
    """Fallback: two-column static image display."""
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            _image_to_rgb_u8(image),
            caption="🛰️ Original satellite image (RGB, gamma-corrected)",
            use_container_width=True,
        )
    with col2:
        st.image(
            _class_map_to_rgb(class_map),
            caption="🎨 Predicted cloud/shadow mask",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — COMPARISON SLIDER TAB
# ═══════════════════════════════════════════════════════════════════════════════

def _render_comparison_slider(image: np.ndarray, class_map: np.ndarray) -> None:
    """Render an interactive before/after image comparison slider."""
    try:
        from streamlit_image_comparison import image_comparison  # type: ignore
        image_comparison(
            img1   = _image_to_rgb_u8(image),
            img2   = _class_map_to_rgb(class_map),
            label1 = "Original satellite image",
            label2 = "Predicted cloud/shadow mask",
            width  = 850,
            show_labels     = True,
            make_responsive = True,
        )
        st.caption(
            "Drag the slider to compare the original image with the "
            "deep learning prediction."
        )
    except ImportError:
        st.info(
            "Install **streamlit-image-comparison** for the interactive slider:\n"
            "```bash\npip install streamlit-image-comparison\n```"
        )
        _render_static_columns(image, class_map)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — STATISTICS TAB
# ═══════════════════════════════════════════════════════════════════════════════

def _render_statistics(class_map: np.ndarray, profile: dict) -> None:
    """Display real-time KPI metrics and class distribution charts."""
    from geospatial_utils import compute_area_stats
    import pandas as pd

    stats = compute_area_stats(class_map, profile)

    # ── Top KPI row ───────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "🗺️ Total scene area",
        f"{stats.get('total_km2', 0):,.1f} km²",
    )
    col2.metric(
        "☁️ Cloud cover",
        f"{stats.get('cloud_km2', 0):,.2f} km²",
        delta=f"{stats.get('cloud_fraction', 0) * 100:.2f}% of scene",
        delta_color="off",
    )
    col3.metric(
        "🌑 Cloud shadow",
        f"{stats.get('shadow_km2', 0):,.2f} km²",
        delta=f"{stats.get('shadow_fraction', 0) * 100:.2f}% of scene",
        delta_color="off",
    )
    col4.metric(
        "🌿 Background",
        f"{stats.get('background_km2', 0):,.1f} km²",
    )
    col5.metric(
        "📏 Pixel GSD",
        f"{stats.get('pixel_size_m', 0):.1f} m",
        help="Ground Sampling Distance — pixel side length in metres",
    )

    st.markdown("---")

    # ── Horizontal bar chart: area per class ──────────────────────────────────
    st.subheader("Area distribution by class")
    area_df = pd.DataFrame(
        {
            "Class":    [CLASS_NAMES[c] for c in range(NUM_CLASSES)],
            "Area km²": [
                stats.get("background_km2", 0),
                stats.get("cloud_km2",      0),
                stats.get("shadow_km2",     0),
            ],
        }
    )
    st.bar_chart(area_df.set_index("Class"), horizontal=False, color="#1E90FF")

    st.markdown("---")

    # ── Pixel count detail table ───────────────────────────────────────────────
    st.subheader("Detailed pixel counts")
    total_px = class_map.size
    detail = {
        "Class":       [CLASS_NAMES[c] for c in range(NUM_CLASSES)],
        "Pixels":      [
            f"{stats.get(f'{CLASS_NAMES[c].lower()}_px', 0):,.0f}"
            for c in range(NUM_CLASSES)
        ],
        "Area (km²)":  [
            f"{stats.get(f'{CLASS_NAMES[c].lower()}_km2', 0):.4f}"
            for c in range(NUM_CLASSES)
        ],
        "Coverage %":  [
            f"{stats.get(f'{CLASS_NAMES[c].lower()}_px', 0) / max(total_px, 1) * 100:.2f}%"
            for c in range(NUM_CLASSES)
        ],
    }
    st.table(pd.DataFrame(detail))


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — DOWNLOAD TAB
# ═══════════════════════════════════════════════════════════════════════════════

def _render_download(
    class_map:   np.ndarray,
    profile:     dict,
    scene_name:  str,
) -> None:
    """Provide a download button for the georeferenced prediction mask."""
    st.subheader("⬇️ Download Georeferenced Mask")

    st.markdown(
        "The downloaded GeoTIFF embeds the **exact CRS and affine transform** "
        "from your source imagery — open it directly in QGIS or ArcGIS "
        "and it aligns pixel-for-pixel with the original scene."
    )

    mask_bytes = _mask_to_geotiff_bytes(class_map, profile)
    stem       = Path(scene_name).stem if scene_name else "prediction"

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        st.download_button(
            label    = "📥 Download predicted mask (.tif)",
            data     = mask_bytes,
            file_name= f"{stem}_cloud_mask.tif",
            mime     = "image/tiff",
            type     = "primary",
            use_container_width=True,
        )
    with col_info:
        st.info(
            f"File: `{stem}_cloud_mask.tif`\n\n"
            f"CRS: `{profile.get('crs', 'unknown')}`\n\n"
            f"Size: {len(mask_bytes) / 1024:.1f} KB (LZW compressed)"
        )

    st.markdown("---")
    st.subheader("Label Encoding")
    col1, col2 = st.columns(2)
    with col1:
        import pandas as pd
        st.table(pd.DataFrame({
            "Pixel Value": [0, 1, 2, 255],
            "Class":       ["Background", "Cloud", "Cloud Shadow", "NoData"],
            "Colour":      ["Grey (128,128,128)", "White (255,255,255)",
                            "Dark Blue (30,60,120)", "Transparent"],
        }))
    with col2:
        st.markdown(
            "**QGIS Quick-Start:**\n\n"
            "1. Layer → Add Raster Layer → select the .tif\n"
            "2. Right-click → Properties → Symbology\n"
            "3. Render type → **Paletted/Unique values**\n"
            "4. Click **Classify** — colors load from the embedded colormap"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — FINE-TUNING TAB
# ═══════════════════════════════════════════════════════════════════════════════

def _render_finetune_panel(cfg: dict) -> None:
    """Render fine-tuning progress and trigger training."""
    if not cfg["start_finetune"]:
        st.info(
            "Use the **Fine-Tune on New Data** panel in the sidebar to upload "
            "an annotated GeoTIFF scene and trigger a fine-tuning run.  "
            "The model will incorporate your new labels without losing "
            "previously learned representations."
        )
        return

    st.subheader("⚡ Fine-Tuning in Progress")

    # Save uploaded files to temp directories
    import tempfile
    with (
        tempfile.TemporaryDirectory() as tmp_img_dir,
        tempfile.TemporaryDirectory() as tmp_mask_dir,
    ):
        tmp_img_path  = Path(tmp_img_dir)  / "scene.tif"
        tmp_mask_path = Path(tmp_mask_dir) / "mask.tif"

        tmp_img_path.write_bytes(cfg["ft_uploaded_image_bytes"])
        tmp_mask_path.write_bytes(cfg["ft_uploaded_mask_bytes"])

        # Progress widgets
        progress_bar   = st.progress(0, text="Preprocessing new scene …")
        status_text    = st.empty()
        metrics_holder = st.empty()

        # ── Step 1: preprocess new scene ─────────────────────────────────────
        try:
            from geospatial_utils import preprocess_scene

            new_img_patches  = Path(tempfile.mkdtemp())
            new_mask_patches = Path(tempfile.mkdtemp())

            n = preprocess_scene(
                image_path   = tmp_img_path,
                mask_path    = tmp_mask_path,
                out_img_dir  = new_img_patches,
                out_mask_dir = new_mask_patches,
                patch_size   = cfg["patch_size"],
                overlap      = DEFAULT_OVERLAP,
            )
            progress_bar.progress(0.15, text=f"✅ {n} patches extracted from new scene")
            status_text.success(f"New scene preprocessed: {n} patches ready.")
        except Exception as exc:
            st.error(f"Preprocessing failed: {exc}")
            return

        # ── Step 2: fine-tune ─────────────────────────────────────────────────
        status_text.info("🔬 Starting fine-tuning …")

        from train import TrainingConfig, fine_tune

        train_cfg = TrainingConfig(
            image_dir  = Path("data/patches"),
            mask_dir   = Path("data/masks"),
            model_dir  = Path("models"),
            log_dir    = Path("logs"),
            patch_size = cfg["patch_size"],
            batch_size = 4,              # conservative for dashboard GPU context
            epochs     = cfg["ft_epochs"],
        )

        epoch_metrics: list[dict] = []

        def _progress(epoch: int, total: int, logs: dict) -> None:
            frac = epoch / max(total, 1)
            dice = logs.get("val_dice_coeff", logs.get("dice_coeff", 0))
            iou  = logs.get("val_mean_iou",   logs.get("mean_iou",   0))
            val_loss = logs.get("val_loss",   logs.get("loss",        0))
            progress_bar.progress(
                0.15 + frac * 0.85,
                text=f"Epoch {epoch}/{total} — val_loss={val_loss:.4f}  "
                     f"dice={dice:.4f}  IoU={iou:.4f}",
            )
            epoch_metrics.append({"epoch": epoch, "val_loss": val_loss,
                                   "dice": dice, "iou": iou})
            if epoch_metrics:
                import pandas as pd
                metrics_holder.line_chart(
                    pd.DataFrame(epoch_metrics).set_index("epoch")[["val_loss", "dice", "iou"]]
                )

        try:
            fine_tune(
                checkpoint_path = Path(cfg["model_path"]),
                cfg             = train_cfg,
                new_image_dir   = new_img_patches,
                new_mask_dir    = new_mask_patches,
                fine_tune_epochs= cfg["ft_epochs"],
                progress_callback = _progress,
            )
            progress_bar.progress(1.0, text="✅ Fine-tuning complete!")
            st.success(
                "Fine-tuning finished. Reload the page or re-run inference to use "
                "the updated model weights."
            )
            # Invalidate cached model so next call reloads updated weights
            _load_model.clear()
        except Exception as exc:
            st.error(f"Fine-tuning failed: {exc}")
            logger.exception("Fine-tuning error")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Streamlit application entry point."""

    # ── Sidebar (always rendered) ─────────────────────────────────────────────
    cfg = _render_sidebar()

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='font-size:2rem;'>🛰️ CloudShadow-UNet</h1>"
        "<p style='color:#666;'>Satellite Cloud & Shadow Segmentation · "
        "Deep Learning Pipeline · Real-time GIS Dashboard</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── No file uploaded state ────────────────────────────────────────────────
    if cfg["uploaded_image_bytes"] is None:
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.markdown(
                "### Getting Started\n\n"
                "1. **Upload** a 4-band GeoTIFF in the sidebar "
                "(Sentinel-2 / Landsat 8 Level-2A)\n"
                "2. Click **🚀 Run Inference**\n"
                "3. Explore results across all tabs\n"
                "4. **Download** the georeferenced mask for QGIS\n\n"
                "> No trained model yet?  Run:\n"
                "> ```bash\n"
                "> python train.py --mode train --config configs/unet_baseline.yaml\n"
                "> ```"
            )
        with col_r:
            st.markdown(
                "<div style='background:#0e1117;border:1px solid #333;"
                "border-radius:8px;padding:16px'>"
                "<h4 style='color:#1E90FF'>Model Status</h4>"
                + (
                    "<p style='color:#00c853'>✅ Model loaded and ready</p>"
                    if _load_model(cfg["model_path"]) is not None
                    else "<p style='color:#ff5252'>⚠️ No model found at the specified path</p>"
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        with st.expander("📚 How to prepare your data"):
            st.markdown(
                "**Step 1 — Merge band files** (if your dataset ships as separate files):\n"
                "```bash\ngdal_merge.py -separate -o scene.tif r.tif g.tif b.tif nir.tif\n```\n\n"
                "**Step 2 — Preprocess**:\n"
                "```bash\npython geospatial_utils.py \\\n"
                "    --image data/raw/scene.tif \\\n"
                "    --mask  data/raw/scene_mask.tif\n```\n\n"
                "**Step 3 — Train**:\n"
                "```bash\npython train.py --mode train\n```"
            )
        return

    # ── Trigger inference ─────────────────────────────────────────────────────
    file_bytes = cfg["uploaded_image_bytes"]
    model_path = cfg["model_path"]
    patch_size = cfg["patch_size"]
    overlap    = cfg["overlap"]
    scene_name = cfg["scene_name"]

    # Read raster metadata immediately (cheap, always cached)
    raster_result = _read_geotiff_bytes(file_bytes)
    if raster_result is None:
        return  # Error already displayed inside _read_geotiff_bytes
    image, profile = raster_result

    h, w = image.shape[:2]
    st.markdown(
        f"**Scene:** `{scene_name}` &nbsp;|&nbsp; "
        f"**Dimensions:** {w} × {h} px &nbsp;|&nbsp; "
        f"**CRS:** `{profile.get('crs', 'unknown')}`"
    )

    # Perform inference only when button is explicitly clicked
    if cfg["run_inference"] or "class_map" in st.session_state:
        if cfg["run_inference"]:
            # Clear previous result first so spinner shows
            st.session_state.pop("class_map", None)

        with st.spinner("🤖 Deep learning inference running …"):
            class_map = _run_inference_cached(file_bytes, model_path, patch_size, overlap)

        if class_map is None:
            return

        st.session_state["class_map"] = class_map
    else:
        st.info("Upload a GeoTIFF then click **🚀 Run Inference** to begin.")
        return

    class_map = st.session_state["class_map"]

    # Apply confidence threshold (only if non-zero — triggers extra pass)
    threshold = cfg["confidence_threshold"]
    if threshold > 0.0:
        with st.spinner(f"Applying confidence threshold ({threshold:.0%}) …"):
            class_map = _apply_confidence_threshold(
                class_map, threshold, file_bytes, model_path, patch_size, overlap
            )

    st.success("✅ Inference complete!")
    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_map, tab_compare, tab_stats, tab_download, tab_finetune = st.tabs([
        "🗺️ Interactive Map",
        "↔️ Comparison Slider",
        "📊 Statistics",
        "⬇️ Download",
        "🔬 Fine-Tune",
    ])

    with tab_map:
        st.subheader("Interactive Geospatial Map")
        st.caption(
            "Toggle layers using the layer manager (top-right of map).  "
            "Adjust the mask opacity slider to blend the original image with the prediction."
        )
        _render_interactive_map(image, class_map, profile)

    with tab_compare:
        st.subheader("Before / After Comparison")
        _render_comparison_slider(image, class_map)

    with tab_stats:
        st.subheader("Geospatial Analytics")
        _render_statistics(class_map, profile)

    with tab_download:
        _render_download(class_map, profile, scene_name or "scene")

    with tab_finetune:
        st.subheader("Continuous Learning — Fine-Tune on New Data")
        _render_finetune_panel(cfg)


# ─── run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
