# CloudShadow-UNet

> **End-to-End Satellite Image Cloud & Shadow Segmentation using Deep Learning**

A production-ready pipeline that uses a custom **U-Net** deep learning architecture to automatically detect and segment **clouds** and **cloud shadows** in multi-spectral satellite imagery (Sentinel-2 / Landsat 8). The system outputs georeferenced prediction masks that open directly in QGIS, ArcGIS, or any GIS software with zero coordinate misalignment — and ships with an interactive **Streamlit dashboard** for real-time inference and geospatial statistics.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Why This Problem Matters](#2-why-this-problem-matters)
3. [Tech Stack](#3-tech-stack)
4. [Project Architecture](#4-project-architecture)
5. [Repository Structure](#5-repository-structure)
6. [Module Breakdown](#6-module-breakdown)
7. [Dataset — 38-Cloud Format](#7-dataset--38-cloud-format)
8. [Installation & Environment Setup](#8-installation--environment-setup)
9. [Step-by-Step Usage Guide](#9-step-by-step-usage-guide)
10. [Configuration Reference](#10-configuration-reference)
11. [Model Architecture Deep Dive](#11-model-architecture-deep-dive)
12. [Loss Function Explained](#12-loss-function-explained)
13. [Inference & Geospatial Output](#13-inference--geospatial-output)
14. [Streamlit Dashboard](#14-streamlit-dashboard)
15. [Environment Variables](#15-environment-variables)
16. [Training Tips & GPU Guide](#16-training-tips--gpu-guide)
17. [Output Label Encoding](#17-output-label-encoding)
18. [Common Errors & Fixes](#18-common-errors--fixes)
19. [Project Roadmap](#19-project-roadmap)
20. [Contributing](#20-contributing)

---

## 1. What This Project Does

CloudShadow-UNet solves a **semantic segmentation** problem:

> Given a raw 16-bit multi-spectral satellite image, classify **every single pixel** into one of three categories:

| Class | Label Value | Meaning |
|---|---|---|
| Background | `0` | Clear terrain, water, vegetation |
| Cloud | `1` | Opaque or thin cirrus cloud cover |
| Cloud Shadow | `2` | Surface darkening caused by clouds blocking sunlight |

The pipeline covers **everything** from raw GeoTIFF ingestion to a web dashboard:

```
Raw GeoTIFF  →  Preprocess  →  Train U-Net  →  Predict  →  Georeferenced Mask  →  Dashboard
```

---

## 2. Why This Problem Matters

Satellites like **Sentinel-2** and **Landsat 8** image the entire Earth every 5–16 days. However, roughly **67% of Earth's surface is cloud-covered at any given time**. Clouds and their shadows corrupt pixel values — making them unusable for:

- Agricultural monitoring (NDVI, crop health)
- Flood and wildfire damage assessment
- Urban change detection
- Ocean colour and sea surface temperature retrieval

**Manual cloud masking** is impossibly slow at satellite scale (terabytes per day). This project provides an automated deep learning solution that achieves state-of-the-art accuracy in seconds per scene.

---

## 3. Tech Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| Deep Learning | TensorFlow 2.x / Keras | U-Net model, training, inference |
| Geospatial I/O | Rasterio | Read/write GeoTIFF with CRS preservation |
| Image Processing | OpenCV | CLAHE contrast enhancement, normalisation |
| Array Operations | NumPy | Tiling, blending, one-hot encoding |
| Coordinate Systems | PyProj | CRS transformation & km² calculation |
| Dashboard | Streamlit | Interactive web UI |
| Map Rendering | Leafmap + Folium | Geospatial interactive maps |
| Configuration | PyYAML | Hyperparameter management |

---

## 4. Project Architecture

```
                     ┌─────────────────────────────────────┐
                     │         RAW GeoTIFF (16-bit)        │
                     │   Bands: Red, Green, Blue, NIR      │
                     └──────────────┬──────────────────────┘
                                    │
                        ┌───────────▼───────────┐
                        │   MODULE 1            │
                        │   Preprocessing       │
                        │   • Normalise [0,1]   │
                        │   • CLAHE per band    │
                        │   • Tile 256×256      │
                        └───────────┬───────────┘
                                    │
                        ┌───────────▼───────────┐
                        │   MODULE 2            │
                        │   Data Generator      │
                        │   • Lazy disk loading │
                        │   • One-hot masks     │
                        │   • Augmentations     │
                        └───────────┬───────────┘
                                    │
                        ┌───────────▼───────────┐
                        │   MODULE 3            │
                        │   U-Net Architecture  │
                        │   • 4-band input      │
                        │   • 4 encoder stages  │
                        │   • Skip connections  │
                        │   • Softmax output    │
                        └───────────┬───────────┘
                                    │
                        ┌───────────▼───────────┐
                        │   MODULE 4            │
                        │   Loss & Training     │
                        │   • Dice Loss 70%     │
                        │   • CCE Loss 30%      │
                        │   • Adam optimizer    │
                        └───────────┬───────────┘
                                    │
                        ┌───────────▼───────────┐
                        │   MODULE 5            │
                        │   Inference           │
                        │   • Sliding window    │
                        │   • Cosine blending   │
                        │   • CRS preservation  │
                        └───────────┬───────────┘
                                    │
                        ┌───────────▼───────────┐
                        │   MODULE 6            │
                        │   Streamlit Dashboard │
                        │   • Leafmap render    │
                        │   • km² statistics    │
                        │   • GeoTIFF download  │
                        └───────────────────────┘
```

---

## 5. Repository Structure

```
CloudShadow-Unet/
│
├── configs/
│   └── unet_baseline.yaml        ← All hyperparameters (epochs, LR, batch size …)
│
├── data/
│   ├── raw/                      ← Place your original GeoTIFF files here (DO NOT modify)
│   ├── patches/                  ← Auto-generated 256×256 image patches (.npy)
│   └── masks/                    ← Auto-generated 256×256 mask patches  (.npy)
│
├── models/
│   ├── best_weights.h5           ← Best checkpoint (saved by ModelCheckpoint)
│   └── final_model.keras         ← Final saved model after training completes
│
├── notebooks/
│   └── 01_explore_dataset.ipynb  ← Visual exploration of scenes and patches
│
├── outputs/
│   └── predicted_mask.tif        ← Georeferenced prediction output
│
├── scripts/
│   ├── download_38cloud.py       ← Automated downloader for 38-Cloud/95-Cloud
│   ├── download_sentinel2.py     ← Downloader for fresh Sentinel-2 imagery
│   └── create_synthetic_demo.py  ← Generates random data for quick testing
│
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py         ← MODULE 1: GeoTIFF reading, CLAHE, tiling
│   │
│   ├── model/
│   │   ├── generator.py          ← MODULE 2: Custom Keras data generator
│   │   ├── unet.py               ← MODULE 3: U-Net architecture
│   │   └── losses.py             ← MODULE 4: Dice Loss, IoU metric, CCE combo
│   │
│   ├── training/
│   │   └── train.py              ← Training orchestrator (CLI entry point)
│   │
│   ├── inference/
│   │   └── predict.py            ← MODULE 5: Sliding window + georeferenced output
│   │
│   └── dashboard/
│       └── app.py                ← MODULE 6: Streamlit web dashboard
│
├── requirements.txt              ← All Python dependencies
├── LICENSE                       ← MIT License
└── README.md                     ← This file
```

---

## 6. Module Breakdown

### Module 1 — `src/preprocessing/preprocess.py`
**What it does:** Converts raw satellite imagery into clean, normalized NumPy arrays ready for deep learning.

**Key functions:**

| Function | Input | Output | Why it exists |
|---|---|---|---|
| `read_multiband_geotiff()` | `.tif` path | `(H,W,4) float32` + profile | Reads 4 bands + preserves CRS metadata |
| `apply_clahe_per_band()` | `(H,W,4) float32` | `(H,W,4) float32` | Enhances thin cirrus clouds in NIR band |
| `generate_patch_coords()` | H, W, patch_size | list of `(r0,r1,c0,c1)` | Avoids partial tiles at image boundaries |
| `tile_image_and_mask()` | image + mask | patch lists | Slices massive arrays for GPU memory |
| `preprocess_scene()` | paths + params | saves `.npy` files | Full preprocessing pipeline |

**Why CLAHE?** Thin cirrus clouds occupy a very narrow slice of the reflectance histogram. CLAHE (Contrast Limited Adaptive Histogram Equalization) redistributes contrast *locally* so the model sees sharper cloud edges without globally amplifying noise.

---

### Module 2 — `src/model/generator.py`
**What it does:** Streams data from disk to GPU memory lazily — never loads the full dataset into RAM.

**Key class:** `CloudSegmentationGenerator(tf.keras.utils.Sequence)`

**Augmentations applied on-the-fly:**
```
Horizontal flip     → 50% probability
Vertical flip       → 50% probability
90° rotation        → random 0°/90°/180°/270°
Brightness jitter   → ±10% uniform offset
Gaussian noise      → σ ≤ 0.01 per pixel
```

> All geometric augmentations are applied **identically** to the image and its mask using a shared random state — so labels never get misaligned.

**Why augment?** Satellite clouds have no fixed orientation relative to the sensor (unlike ground-level photos). Rotational augmentation forces the model to learn orientation-invariant features.

---

### Module 3 — `src/model/unet.py`
**What it does:** Defines the U-Net model that performs pixel-wise segmentation.

**Architecture at a glance:**
```
Input (B, 256, 256, 4)
    │
    ├── Encoder Block 1  → 64  filters → skip1  → MaxPool → (B,128,128,64)
    ├── Encoder Block 2  → 128 filters → skip2  → MaxPool → (B,64,64,128)
    ├── Encoder Block 3  → 256 filters → skip3  → MaxPool → (B,32,32,256)
    ├── Encoder Block 4  → 512 filters → skip4  → MaxPool → (B,16,16,512)
    │
    ├── Bottleneck       → 1024 filters              (B,16,16,1024)
    │
    ├── Decoder Block 4  → Upsample + Concat(skip4) → 512 filters
    ├── Decoder Block 3  → Upsample + Concat(skip3) → 256 filters
    ├── Decoder Block 2  → Upsample + Concat(skip2) → 128 filters
    ├── Decoder Block 1  → Upsample + Concat(skip1) → 64  filters
    │
    └── Output Conv 1×1 → softmax → (B, 256, 256, 3)
```

**Total parameters: ~31 million** (fits on a single 8 GB GPU with batch_size=8)

---

### Module 4 — `src/model/losses.py`
**What it does:** Provides the custom loss function and metrics that handle class imbalance.

See [Loss Function Explained](#12-loss-function-explained) for the full mathematical breakdown.

---

### Module 5 — `src/inference/predict.py`
**What it does:** Runs a trained model on a full-scene GeoTIFF and writes a georeferenced output mask.

**Key algorithm — Cosine-Bell Blending:**
```
For each overlapping tile:
    1. Extract patch from padded image
    2. Predict softmax probabilities (B, H, W, 3)
    3. Multiply predictions by cosine-bell weight mask
       (centre weight ≈ 1.0, edge weight ≈ 0.0)
    4. Accumulate weighted predictions into full-scene array
    5. Divide accumulated sum by weight sum → blended probabilities
    6. Argmax → class label map
```

This eliminates the grid artefacts that appear when tile predictions are naively stitched together.

---

### Module 6 — `src/dashboard/app.py`
**What it does:** A Streamlit web application for drag-and-drop cloud segmentation.

See [Streamlit Dashboard](#14-streamlit-dashboard) for full feature list.

---

## 7. Dataset — 38-Cloud Format

This project is designed to work with the **[38-Cloud Dataset](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset)** and any dataset following its conventions.

### Input Image Format
- **Type:** GeoTIFF (`.tif`)
- **Bit depth:** 16-bit unsigned integer (`uint16`)
- **Bands:** 4 (Red, Green, Blue, NIR in that order)
- **Reflectance scale:** Raw values divided by `10,000` to get surface reflectance in `[0, 1]`
- **CRS:** UTM (typical) — e.g., EPSG:32632

### Ground Truth Mask Format
- **Type:** GeoTIFF (`.tif`) or PNG
- **Bit depth:** 8-bit unsigned integer (`uint8`)
- **Values:** `0` = Background, `1` = Cloud, `2` = Cloud Shadow
- **CRS:** Must match the image CRS exactly

### How to Prepare Your Data

```
data/
└── raw/
    ├── scene_001.tif           ← 4-band image
    ├── scene_001_mask.tif      ← Corresponding ground truth mask
    ├── scene_002.tif
    ├── scene_002_mask.tif
    └── ...
```

> If using the 38-Cloud dataset, it ships with separate R/G/B/NIR band files. Merge them first using GDAL:
>
> ```bash
> gdal_merge.py -separate -o scene_001.tif red.tif green.tif blue.tif nir.tif
> ```

---

## 8. Installation & Environment Setup

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9 – 3.11 |
| CUDA (for GPU training) | 11.8 or 12.x |
| cuDNN | 8.6+ |
| GDAL (system library) | 3.6+ |

### Step 1 — Clone the Repository

```bash
git clone https://github.com/pravin-python/CloudShadow-Unet.git
cd CloudShadow-Unet
```

### Step 2 — Create a Virtual Environment

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# OR using conda (recommended for GDAL)
conda create -n cloudseg python=3.10
conda activate cloudseg
conda install -c conda-forge gdal rasterio
```

### Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Verify GPU is Detected (Optional)

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Should print: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Step 5 — Verify GDAL / Rasterio

```bash
python -c "import rasterio; print(rasterio.__version__)"
# Should print: 1.3.x or higher
```

---

## 9. Step-by-Step Usage Guide

Follow these steps in order. Each step feeds into the next.

---

### Step 1 — Get Your Data

You can either place your own data manually or use the automated download scripts:

**A. Automated Download (38-Cloud / 95-Cloud)**
Recommended for beginners. Requires a Kaggle account.
```bash
# Downloads 95-Cloud (includes cloud + shadow labels)
python scripts/download_38cloud.py --source kaggle --dataset 95cloud
```

**B. Automated Download (Sentinel-2)**
Downloads fresh imagery from Copernicus Data Space. Requires a free [CDSE account](https://dataspace.copernicus.eu/).
```bash
python scripts/download_sentinel2.py --username YOUR_USER --password YOUR_PASS --bbox "lon_min,lat_min,lon_max,lat_max" --date_start 2024-01-01 --date_end 2024-01-30
```

**C. Manual Placement**
Place your 4-band GeoTIFFs (Red, Green, Blue, NIR) and their corresponding masks here:
```
data/raw/
├── scene_001.tif          ← 4-band GeoTIFF
├── scene_001_mask.tif     ← Ground truth mask
├── scene_002.tif
└── scene_002_mask.tif
```

---

### Step 2 — Preprocess All Scenes

Run preprocessing on each scene to generate 256×256 NumPy patch files:

```bash
python src/preprocessing/preprocess.py \
    --image  data/raw/scene_001.tif \
    --mask   data/raw/scene_001_mask.tif \
    --out_img  data/patches \
    --out_mask data/masks \
    --patch_size 256 \
    --overlap 0.25
```

**What gets created:**
```
data/patches/scene_001_patch_00000.npy    ← float32 (256,256,4)
data/patches/scene_001_patch_00001.npy
...
data/masks/scene_001_patch_00000.npy      ← uint8 (256,256)
data/masks/scene_001_patch_00001.npy
...
```

To preprocess multiple scenes at once:
```bash
for f in data/raw/*[!mask].tif; do
    stem="${f%.*}"
    python src/preprocessing/preprocess.py \
        --image "$f" \
        --mask "${stem}_mask.tif"
done
```

---

### Step 3 — Explore the Dataset (Optional)

```bash
jupyter notebook notebooks/01_explore_dataset.ipynb
```

This notebook shows:
- RGB and NIR band visualisation
- Ground truth mask with colour coding
- Before/after CLAHE comparison
- Sample patch grid

---

### Step 4 — Train the Model

```bash
python src/training/train.py --config configs/unet_baseline.yaml
```

**What happens during training:**
```
Epoch 1/100
    → Loads batches from data/patches + data/masks via generator
    → Applies random augmentations (flip, rotate, brightness, noise)
    → Forward pass through U-Net
    → Computes Combined Dice+CCE Loss
    → Backprop + Adam update
    → Logs val_loss, val_dice_coeff, val_mean_iou

After each epoch:
    → ModelCheckpoint: saves models/best_weights.h5 if val_loss improved
    → ReduceLROnPlateau: halves LR if val_loss stalls for 5 epochs
    → EarlyStopping: stops training if val_loss stalls for 15 epochs
    → TensorBoard: updates logs/
    → CSVLogger: appends to logs/training_log.csv
```

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir logs/
# Open: http://localhost:6006
```

**Expected training time:**

| Hardware | 1000 patches, 100 epochs |
|---|---|
| NVIDIA RTX 3090 (24 GB) | ~2 hours |
| NVIDIA RTX 3060 (12 GB) | ~4 hours |
| Apple M2 Pro (MPS) | ~6 hours |
| CPU only | ~24+ hours (not recommended) |

---

### Step 5 — Run Inference on a New Scene

```bash
python src/inference/predict.py \
    --input  data/raw/new_scene.tif \
    --output outputs/predicted_mask.tif \
    --model  models/best_weights.h5 \
    --patch_size 256 \
    --overlap 0.25
```

**What gets created:**
```
outputs/predicted_mask.tif    ← uint8 GeoTIFF with:
                                  Band 1: class labels (0/1/2)
                                  CRS: copied from source
                                  Transform: copied from source
```

**Terminal output shows:**
```
Class 0 (Background): 1,823,412 px — 87.34 %
Class 1 (Cloud):         201,203 px — 9.64 %
Class 2 (Shadow):         62,201 px — 2.98 %
background_km2:   182.34
cloud_km2:         20.12
shadow_km2:         6.22
total_scene_km2:  208.68
cloud_fraction:     0.096
shadow_fraction:    0.029
```

---

### Step 6 — Launch the Dashboard

```bash
streamlit run src/dashboard/app.py
```

Opens at: **http://localhost:8501**

---

### Step 7 — Load Result in QGIS

1. Open QGIS
2. **Layer → Add Layer → Add Raster Layer**
3. Select `outputs/predicted_mask.tif`
4. The mask aligns pixel-perfectly with your source imagery because CRS and affine transform are preserved

---

## 10. Configuration Reference

All training hyperparameters live in `configs/unet_baseline.yaml`:

```yaml
# Data
patch_size: 256          # Tile size in pixels — must match preprocessing
overlap: 0.25            # Sliding window overlap
val_fraction: 0.15       # 15% of patches held out for validation
seed: 42                 # Random seed for reproducibility

# Model
num_classes: 3           # Never change — Background, Cloud, Shadow
base_filters: 64         # Filters in first encoder block (doubles each level)
depth: 4                 # Encoder/decoder stages
dropout_rate: 0.10       # Dropout in conv blocks
bottleneck_dropout: 0.30 # Higher dropout in bottleneck

# Training
epochs: 100
batch_size: 8            # Increase to 16 if GPU VRAM > 16 GB
learning_rate: 0.0001    # Adam initial LR
dice_alpha: 0.70         # Dice weight (1-alpha = CCE weight)

# Callbacks
reduce_lr_patience: 5    # Epochs before LR is halved
early_stop_patience: 15  # Epochs before early stopping

# System
mixed_precision: false   # Set true for RTX 30xx / A100 GPUs
workers: 4               # CPU threads for data loading
use_multiprocessing: true
```

### Tuning Guide

| Goal | Parameter to change |
|---|---|
| Out of GPU memory | Reduce `batch_size` to 4, or `patch_size` to 128 |
| Faster training | Enable `mixed_precision: true` on Ampere+ GPUs |
| Better shadow detection | Increase `dice_alpha` to 0.85 |
| Less overfitting | Increase `dropout_rate` to 0.2 |
| More model capacity | Increase `base_filters` to 96 or 128 |

---

## 11. Model Architecture Deep Dive

### Why U-Net?

U-Net was specifically designed for biomedical image segmentation where:
- Training data is limited
- Precise pixel-level boundaries matter
- Objects appear at multiple scales

These same properties apply perfectly to cloud segmentation — clouds appear from tiny wisps to continent-spanning systems, and their edges must be sharp.

### Skip Connections — The Critical Innovation

```
Encoder level 3: [256 feature maps, full spatial detail]
         ↓ MaxPool (loses spatial detail)
Bottleneck: [1024 feature maps, compressed semantics]
         ↓ Upsample (recovers spatial resolution)
Decoder level 3: [Concat encoder + decoder] → fuses detail + semantics
```

Without skip connections, upsampled predictions are blurry. Skip connections inject the fine spatial detail from the encoder directly into the decoder — producing sharp cloud boundaries.

### SpatialDropout2D vs Dropout

Regular `Dropout` kills individual pixels randomly. Adjacent pixels in a feature map are highly correlated — so killing one pixel barely changes the map.

`SpatialDropout2D` drops **entire feature maps** at once, forcing the remaining maps to learn independent representations. This is significantly more effective for convolutional features.

### Conv2DTranspose vs Bilinear Upsampling

| Method | Parameters | Learns upsampling? |
|---|---|---|
| Bilinear + Conv2D | ~0 + (k×k×C_in×C_out) | Partially |
| Conv2DTranspose | (k×k×C_in×C_out) | Yes, fully |

`Conv2DTranspose` (a.k.a. deconvolution) learns its own upsampling kernel, which is especially important for irregular shapes like cloud edges.

---

## 12. Loss Function Explained

### The Class Imbalance Problem

In a typical Sentinel-2 scene:
```
Background:    ~87% of pixels
Cloud:          ~9% of pixels
Shadow:         ~3% of pixels
```

If you use standard **Categorical Cross-Entropy** on this data, the model learns that "predict everything as Background" gives ~87% pixel accuracy — and it stops learning to detect clouds or shadows.

### Dice Loss — The Solution

The Dice Coefficient measures **overlap** between prediction and ground truth:

```
Dice(y_true, y_pred) = (2 × |y_true ∩ y_pred| + ε) / (|y_true| + |y_pred| + ε)
```

Dice Loss = `1 - Dice Coefficient`

**Key property:** Dice Loss is scale-invariant. Whether a class has 10 pixels or 10 million pixels, the contribution to the loss is the same — every class matters equally.

### Multi-Class Dice Loss

For 3 classes, compute Dice Loss per class then average:

```
MultiDiceLoss = mean(DiceLoss_Background, DiceLoss_Cloud, DiceLoss_Shadow)
```

### Combined Loss (Used in Training)

```
CombinedLoss = 0.70 × MultiDiceLoss + 0.30 × CategoricalCrossEntropy
```

- **70% Dice:** Handles class imbalance — never lets background dominate
- **30% CCE:** Provides pixel-level gradient signal during early training when predictions are near-uniform (Dice gradient is weak near 0.5)

### Custom Metrics

| Metric | Formula | Range | Higher = Better |
|---|---|---|---|
| Dice Coefficient | `2|TP| / (2|TP| + |FP| + |FN|)` | [0, 1] | Yes |
| Mean IoU | `|TP| / (|TP| + |FP| + |FN|)` | [0, 1] | Yes |

Both are computed as epoch-level running averages (not batch averages) for unbiased evaluation.

---

## 13. Inference & Geospatial Output

### Sliding Window Algorithm

A full Sentinel-2 scene can be 10,980 × 10,980 pixels — far too large to fit in GPU memory. The sliding window breaks it into 256×256 tiles with 25% overlap.

```
Image: 10980×10980
Stride: 256 × (1-0.25) = 192 pixels
Number of tiles: ((10980-256)/192 + 1)² ≈ 3,136 tiles
```

### Cosine-Bell Blending

Without blending, tile boundaries create visible grid artefacts in the output mask. The cosine-bell weight mask (Hanning window) assigns:
- Weight **1.0** at tile centre (most reliable prediction)
- Weight **~0.0** at tile edges (most unreliable prediction)

Overlapping tiles are accumulated with their weights and divided by the total weight sum — producing a smooth, seamless output.

### CRS Preservation — Critical for QGIS

```python
# Source profile (from input GeoTIFF):
profile = {
    'crs':       CRS.from_epsg(32632),          # UTM Zone 32N
    'transform': Affine(10.0, 0.0, 399960.0,   # 10m pixel size
                        0.0, -10.0, 5300040.0),  # Top-left corner
    'width':  10980,
    'height': 10980,
    ...
}

# Output mask inherits this profile verbatim:
# → Same CRS  → same coordinate system
# → Same transform → same pixel↔coordinate mapping
# Result: mask aligns perfectly when dragged into QGIS
```

---

## 14. Streamlit Dashboard

### Features

| Tab | Feature |
|---|---|
| **Interactive Map** | Leafmap view with RGB image and mask as toggleable layers |
| **Image Comparison** | Before/after slider (requires `streamlit-image-comparison`) |
| **Statistics** | Cloud km², shadow km², scene total km², cloud fraction |
| **Download** | Georeferenced GeoTIFF mask ready for QGIS |

### Caching Strategy

```python
# Model: loaded ONCE per server process, never reloaded on widget interaction
@st.cache_resource
def load_model(model_path: str): ...

# Raster data: cached per file content hash — re-uploading same file = no re-read
@st.cache_data
def read_uploaded_geotiff(file_bytes: bytes): ...

# Inference: cached per (file_bytes, model_path) — switching tabs = no re-inference
@st.cache_data
def run_inference_cached(image, model_path): ...
```

Without these caches, uploading a large GeoTIFF would reload the model and re-run inference on every Streamlit widget interaction — causing Out-of-Memory crashes.

### Running the Dashboard

```bash
streamlit run src/dashboard/app.py

# With custom port:
streamlit run src/dashboard/app.py --server.port 8080

# Allow external access (e.g., on a remote server):
streamlit run src/dashboard/app.py --server.address 0.0.0.0
```

---

## 15. Environment Variables

Override default paths without modifying code:

```bash
export DATA_DIR=/mnt/ssd/satellite_data    # Root data directory
export MODEL_PATH=/mnt/models/unet.h5      # Model weights path
export PATCH_SIZE=256                       # Inference tile size
export OVERLAP=0.25                         # Sliding window overlap
export NUM_CLASSES=3                        # Segmentation classes
```

Or create a `.env` file and load it:
```bash
# .env
DATA_DIR=data
MODEL_PATH=models/best_weights.h5
PATCH_SIZE=256
OVERLAP=0.25
NUM_CLASSES=3
```

---

## 16. Training Tips & GPU Guide

### GPU Memory Requirements

| Patch Size | Batch Size | Min VRAM |
|---|---|---|
| 128×128 | 16 | 4 GB |
| 256×256 | 8 | 8 GB |
| 256×256 | 16 | 14 GB |
| 384×384 | 8 | 16 GB |
| 512×512 | 8 | 24 GB |

### Mixed Precision Training (Ampere GPUs — RTX 30xx / A100 / H100)

Enable in `configs/unet_baseline.yaml`:
```yaml
mixed_precision: true
```

Effect: Forward/backward passes run in `float16` (2× faster, 2× less VRAM). Weights stored in `float32` (no accuracy loss). The output `Conv2D` layer is forced to `float32` to maintain softmax numerical stability.

### Multi-GPU Training

Wrap model creation in a `MirroredStrategy`:
```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_unet(...)
    model.compile(...)
```

Then increase `batch_size` proportionally to number of GPUs.

### Early Signs of Good Training

```
Epoch 1:  val_loss=0.82  val_dice_coeff=0.31  val_mean_iou=0.19
Epoch 10: val_loss=0.54  val_dice_coeff=0.61  val_mean_iou=0.44
Epoch 30: val_loss=0.38  val_dice_coeff=0.74  val_mean_iou=0.59
Epoch 50: val_loss=0.28  val_dice_coeff=0.82  val_mean_iou=0.69
```

### Signs of Problems

| Symptom | Likely Cause | Fix |
|---|---|---|
| `val_dice_coeff` stuck at 0.0 | Label encoding wrong | Check masks are `0/1/2`, not `0/128/255` |
| Loss spikes after epoch 1 | LR too high | Reduce `learning_rate` to `1e-5` |
| OOM error | Batch too large | Reduce `batch_size` |
| No improvement after 20 epochs | Too few patches | Preprocess more scenes |

---

## 17. Output Label Encoding

The predicted GeoTIFF mask uses these pixel values:

| Pixel Value | Class | Display Colour | RGB |
|---|---|---|---|
| `0` | Background | Grey | (128, 128, 128) |
| `1` | Cloud | White | (255, 255, 255) |
| `2` | Cloud Shadow | Dark Blue | (30, 60, 120) |
| `255` | NoData | — | — |

**Apply colour map in QGIS:**
1. Right-click layer → Properties → Symbology
2. Render type: **Paletted/Unique values**
3. Classify → assign colours manually

---

## 18. Common Errors & Fixes

### `ValueError: X has only 3 band(s); 4 required`
Your GeoTIFF does not have a NIR band. The model requires 4 bands (R, G, B, NIR). Merge NIR band using GDAL:
```bash
gdal_merge.py -separate -o merged.tif red.tif green.tif blue.tif nir.tif
```

### `FileNotFoundError: No .npy files found in data/patches`
You skipped the preprocessing step. Run:
```bash
python src/preprocessing/preprocess.py --image data/raw/scene.tif --mask data/raw/scene_mask.tif
```

### `ResourceExhaustedError: OOM when allocating tensor`
GPU ran out of memory. Reduce batch size in `configs/unet_baseline.yaml`:
```yaml
batch_size: 4
```

### `ImportError: No module named 'rasterio'`
GDAL is not installed at the system level. On macOS:
```bash
brew install gdal
pip install rasterio
```
On Ubuntu/Debian:
```bash
sudo apt-get install gdal-bin libgdal-dev
pip install rasterio
```

### Mask does not align in QGIS
The source GeoTIFF did not have a valid CRS. Assign one:
```bash
gdal_translate -a_srs EPSG:32632 input.tif output_with_crs.tif
```

### Dashboard crashes with OOM when uploading large file
Add `--server.maxUploadSize` flag:
```bash
streamlit run src/dashboard/app.py --server.maxUploadSize 2048
```

### `error: Microsoft Visual C++ 14.0 or greater is required` (Windows)
This occurs when installing `GDAL` via pip on Windows. 
**Fix:** You do NOT need the standalone `GDAL` python package on Windows because `rasterio` bundles its own GDAL binaries. 
1. Open `requirements.txt`.
2. Comment out the `GDAL==3.8.4` line.
3. Run `pip install -r requirements.txt` again.

---

## 19. Project Roadmap

- [x] Module 1: Geospatial Data Preprocessing (Rasterio + OpenCV)
- [x] Module 2: Custom TensorFlow Data Generator
- [x] Module 3: Multi-Class U-Net Architecture
- [x] Module 4: Dice Loss + Metrics
- [x] Module 5: Sliding-Window Inference + Georeferenced Output
- [x] Module 6: Streamlit Dashboard with Leafmap
- [x] Sentinel-2 Level-1C → Level-2A atmospheric correction integration
- [x] Model export to ONNX for cross-framework deployment
- [x] Ensemble inference (multiple checkpoint averaging)
- [x] Active learning loop for iterative dataset expansion
- [x] REST API wrapper (FastAPI) for programmatic access
- [x] Docker container for one-command deployment
- [x] Benchmark against FMask, Sen2Cor, and s2cloudless

---

## 20. Contributing

Contributions are welcome.

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make changes — follow the code standards in CLAUDE.md
# 4. Test your changes
python -m pytest tests/

# 5. Open a Pull Request
```

**Code standards:**
- Python: PEP 8, snake_case, type hints on all public functions
- Docstrings on all public functions (Google style)
- No silent `except: pass` blocks
- Use `pathlib.Path` over `os.path`
- Use `rasterio` for all GeoTIFF I/O — never strip spatial metadata

---

## Quick Reference Card

```bash
# INSTALL
pip install -r requirements.txt

# PREPROCESS a scene
python src/preprocessing/preprocess.py \
    --image data/raw/scene.tif --mask data/raw/scene_mask.tif

# TRAIN
python src/training/train.py --config configs/unet_baseline.yaml

# MONITOR training
tensorboard --logdir logs/

# PREDICT on a new scene
python src/inference/predict.py \
    --input data/raw/new_scene.tif \
    --output outputs/mask.tif \
    --model models/best_weights.h5

# DASHBOARD
streamlit run src/dashboard/app.py
```

---

*Built with TensorFlow, Rasterio, OpenCV, and Streamlit.*
*Designed for Sentinel-2 and Landsat 8 Level-2A imagery.*
