s# CLAUDE.md — Project Instructions

## Project Overview

This is the **CloudShadow-UNet** project: an advanced satellite image cloud and shadow segmentation pipeline using deep learning (U-Net architecture), TensorFlow, OpenCV, NumPy, Rasterio, QGIS, and Streamlit.

The project involves:
- Multi-band GeoTIFF ingestion and preprocessing (Sentinel-2 / Landsat 8)
- Custom U-Net architecture for pixel-wise multi-class segmentation
- Specialized loss functions (Multi-Class Dice Loss) for class imbalance
- Geospatial post-processing with coordinate reference system preservation
- Interactive Streamlit dashboard for real-time inference

---

## AI Behavior Rules

### General
- Be precise and technical — the user understands deep learning and geospatial data
- Prefer working code over explanations; show, don't just tell
- Default to Python unless another language is explicitly requested
- Use type hints in Python; use TypeScript in any JS/TS context
- Never hardcode credentials, API keys, or file paths — use environment variables

### Code Standards
- Python: PEP 8, snake_case, type hints, docstrings on public functions
- Always handle exceptions — no silent `except: pass` blocks
- Use `pathlib.Path` over `os.path` for file operations
- Prefer `numpy` vectorized operations over Python loops for array work
- Use `rasterio` for all GeoTIFF read/write — never strip spatial metadata

### ML / Deep Learning Specifics
- TensorFlow 2.x with Keras API (`tf.keras`)
- Default batch size: 8–16 (adjust for GPU memory)
- Default patch size: 256×256 pixels
- Loss function: Multi-Class Dice Loss (not categorical cross-entropy alone)
- Metrics: IoU (Jaccard Index) and Dice Coefficient — both required
- Model checkpointing: save best weights by `val_loss`
- Mixed precision: use `tf.keras.mixed_precision` when GPU memory is constrained

---

## Skills — When to Load

Skills are in `/skills/`. Load a skill when the task matches its domain.

| Task Type | Load Skill |
|---|---|
| Build or improve any UI, dashboard, or frontend | `frontend-design` |
| Improve user flow, onboarding, or UX quality | `ui-ux-pro-max` |
| SEO, meta tags, rankings, technical SEO | `seo` |
| Review, refactor, or audit code quality | `code-review` |
| Create programmatic videos with Remotion | `remotion` |
| Write secure code, prevent vulnerabilities, OWASP | `owasp-security` |

**Pairing rules:**
- UI tasks: load `frontend-design` + `ui-ux-pro-max` together
- Code with security concerns: load `code-review` + `owasp-security` together
- Never load a skill that isn't relevant to the current task

---

## Project-Specific Constraints

1. **Geospatial metadata must always be preserved.** When writing output GeoTIFFs, copy the source raster profile via `rasterio` — never output a raw numpy array without CRS and affine transform.

2. **Memory safety.** Large rasters must be processed via sliding window. Never load a full-scene raster into GPU memory. Default tile size: 256×256 with 25% overlap.

3. **Class imbalance.** The dataset has extreme imbalance (Background >> Cloud > Shadow). Always use Dice Loss or Focal Loss — never standard categorical cross-entropy alone.

4. **Four-band input.** Model input is always (R, G, B, NIR) — never RGB-only. NIR is essential for distinguishing water from cloud shadow.

5. **QGIS compatibility.** All output masks must be importable into QGIS without coordinate misalignment. Test with `rasterio.open()` to verify CRS is intact before considering a task complete.

6. **Streamlit dashboard caching.** Use `@st.cache_resource` for model loading and `@st.cache_data` for data preprocessing. Never reload the model on every interaction.

---

## File & Folder Conventions

```
CloudShadow-Unet/
├── data/
│   ├── raw/           ← Original GeoTIFF files (do not modify)
│   ├── patches/       ← 256×256 extracted patches (numpy .npy format)
│   └── masks/         ← Corresponding ground truth masks
├── models/            ← Saved model weights (.h5 or SavedModel)
├── outputs/           ← Inference results (predicted GeoTIFF masks)
├── src/
│   ├── preprocessing/ ← Data loading, tiling, normalization
│   ├── model/         ← U-Net architecture, loss functions, metrics
│   ├── training/      ← Training loop, data generators, callbacks
│   ├── inference/     ← Sliding window, blending, georeferencing
│   └── dashboard/     ← Streamlit app
├── skills/            ← AI skills (SKILL.md files)
├── CLAUDE.md          ← This file
└── settings.json      ← AI environment settings
```

---

## Quick Reference

**Run training:**
```bash
python src/training/train.py --config configs/unet_baseline.yaml
```

**Run inference on a scene:**
```bash
python src/inference/predict.py --input data/raw/scene.tif --output outputs/mask.tif
```

**Launch Streamlit dashboard:**
```bash
streamlit run src/dashboard/app.py
```

**Key environment variables:**
```
DATA_DIR        — Path to data directory
MODEL_PATH      — Path to trained model weights
PATCH_SIZE      — Tile size for sliding window (default: 256)
OVERLAP         — Sliding window overlap fraction (default: 0.25)
NUM_CLASSES     — Number of segmentation classes (default: 3)
```
