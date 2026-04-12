### Summary
Implemented a comprehensive pytest suite and resolved bugs in the training metrics to prepare the data and modeling pipeline. We also removed physically-invalid augmentations from the data pipeline and fixed environment build errors.

### Problem Addressed
1. Missing unit tests for critical data pipeline and mathematical functions (loss, metrics, stitching).
2. Bug in Keras metrics `DiceCoefficient` and `MeanIoU` where `add_weight` received an invalid `shape` argument.
3. Use of `A.RandomBrightnessContrast` and similar augmentations in `dataset.py` that destroyed the physical reflectance signatures of clouds and water bodies.
4. Python `GDAL` package installation failure due to a mismatch with the system library version.

### Technical Implementation
- Removed `A.RandomBrightnessContrast`, `A.GaussNoise`, and `A.GaussianBlur` from `_build_albumentations_pipeline` in `dataset.py`.
- Fixed `DiceCoefficient` and `MeanIoU` in `model.py` by properly using the `name` kwargs in `add_weight`.
- Downgraded `GDAL` in `requirements.txt` to `3.8.4` to align with the system-provided `libgdal-dev` version.
- Added a full pytest suite under `tests/` covering `dataset.py`, `geospatial_utils.py`, and `model.py`.

### Validation (tests, metrics)
- Added `test_dataset.py` to ensure `CloudPatchDataset` correctly reads `.npy` arrays, maintains batch shapes, and properly instantiates train/val splits.
- Added `test_model.py` to mathematically validate `MultiClassDiceLoss`, `CombinedDiceCELoss`, `DiceCoefficient`, and `MeanIoU` against expected tensor operations.
- Added `test_geospatial.py` to validate `generate_tile_coords`, `cosine_bell_mask`, and the sliding window `stitch_predictions` logic.
- All tests pass (`pytest tests/`).

### Impact
- **Accuracy**: Ensures physics-based integrity by removing spectral-altering augmentations.
- **Stability**: Prevents runtime crashes from invalid metrics initialization and provides automated verification for future changes.
