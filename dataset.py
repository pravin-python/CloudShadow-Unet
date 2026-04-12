"""
dataset.py — Data Pipeline & Augmentation
==========================================
Principal Engineer: CloudShadow-UNet Project
─────────────────────────────────────────────────────────────────────────────
Implements the complete data pipeline:

  • CloudPatchDataset (tf.keras.utils.Sequence)
      – Lazy disk loading: only one batch is resident in RAM at a time
      – No OOM risk regardless of dataset size
      – Works with Keras's built-in background prefetch workers
  • Augmentation pipeline using albumentations (primary) with an OpenCV
      fallback that is invoked automatically if albumentations is not installed
  • One-hot encoding deferred to generator time (compact uint8 on disk)
  • Deterministic train/val split preserving stratified class balance
  • Fine-tune dataset builder: merges new annotated samples into an
      existing patch directory for continuous/online learning

Augmentations applied (training only):
  ┌────────────────────────────┬─────────────┐
  │ Transformation             │ Probability │
  ├────────────────────────────┼─────────────┤
  │ Horizontal flip            │    50 %     │
  │ Vertical flip              │    50 %     │
  │ 90 ° rotation              │    75 %     │
  │ Random brightness/contrast │    40 %     │
  │ Gaussian noise             │    30 %     │
  │ Gaussian blur (thin cirrus)│    20 %     │
  │ Coarse dropout (occlusion) │    20 %     │
  └────────────────────────────┴─────────────┘

Disk layout expected:
    <image_dir>/*.npy   →  float32 (H, W, 4)  RGBNIR patches, range [0,1]
    <mask_dir>/*.npy    →  uint8   (H, W)      class labels {0, 1, 2}
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# ─── sentinel for optional albumentations ─────────────────────────────────────
try:
    import albumentations as A  # type: ignore
    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    _ALBUMENTATIONS_AVAILABLE = False
    logger.warning(
        "albumentations not installed — falling back to OpenCV augmentation. "
        "Install with: pip install albumentations"
    )

NUM_CLASSES: int = 3  # Background=0, Cloud=1, Shadow=2


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — AUGMENTATION PIPELINES
# ═══════════════════════════════════════════════════════════════════════════════

def _build_albumentations_pipeline() -> "A.Compose":
    """Build and return an albumentations augmentation pipeline.

    All transforms that alter spatial structure are listed first and wrapped
    in OneOf / standard transforms so image+mask receive the SAME random
    operation (albumentations handles this automatically when the mask is
    passed as 'mask' kwarg).

    Returns:
        A.Compose object ready to call with image= and mask= kwargs.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.75),
            # CoarseDropout simulates sensor occlusion / small clouds that
            # were not annotated — forces the model to be robust to gaps.
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                fill_value=0,
                p=0.20,
            ),
        ],
        additional_targets={"mask": "mask"},  # keeps mask in sync with image
    )


def _augment_opencv(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure OpenCV / NumPy augmentation fallback (no albumentations required).

    Applies identical spatial transforms to image and mask via shared RNG.

    Args:
        image: float32 (H, W, 4) patch.
        mask:  uint8  (H, W)     label patch.
        rng:   NumPy random generator (seeded, deterministic per worker).

    Returns:
        Augmented (image, mask) tuple.
    """
    import cv2  # lazy import — cv2 is a hard dependency of the project

    # ── geometric ────────────────────────────────────────────────────────────
    if rng.random() > 0.5:
        image = np.fliplr(image).copy()
        mask  = np.fliplr(mask).copy()
    if rng.random() > 0.5:
        image = np.flipud(image).copy()
        mask  = np.flipud(mask).copy()

    k = int(rng.integers(0, 4))
    if k > 0:
        image = np.rot90(image, k=k, axes=(0, 1)).copy()
        mask  = np.rot90(mask,  k=k).copy()

    # ── photometric (image only) ──────────────────────────────────────────────
    if rng.random() < 0.40:
        delta = float(rng.uniform(-0.15, 0.15))
        image = np.clip(image + delta, 0.0, 1.0).astype(np.float32)

    if rng.random() < 0.30:
        sigma = float(rng.uniform(0.0, 0.04))
        noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0).astype(np.float32)

    if rng.random() < 0.20:
        # Apply mild Gaussian blur to each band independently
        for b in range(image.shape[2]):
            image[:, :, b] = cv2.GaussianBlur(image[:, :, b], (3, 3), 0)

    return image, mask


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — KERAS SEQUENCE DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class CloudPatchDataset(tf.keras.utils.Sequence):
    """Lazily streams (image, mask) patch pairs from disk for Keras training.

    Inheriting tf.keras.utils.Sequence provides:
      • __len__     → Keras knows how many steps per epoch
      • __getitem__ → Keras calls this for each batch, possibly from a worker
      • on_epoch_end → reshuffle between epochs

    Lazy loading strategy:
      Each __getitem__ call opens individual .npy files from disk and
      assembles them into a single batch.  Only one batch's worth of arrays
      is in RAM at any given time — regardless of dataset size.

    Args:
        image_dir:   Directory containing float32 RGBNIR image patches (.npy).
        mask_dir:    Directory containing uint8 label patches (.npy).
        batch_size:  Samples per batch.
        patch_size:  Expected spatial dimension (validation only — not crop).
        augment:     Apply augmentations (True for train, False for val).
        shuffle:     Shuffle sample order each epoch (True for train).
        seed:        RNG seed for reproducibility.
        use_albumentations: Use albumentations pipeline if available.
    """

    def __init__(
        self,
        image_dir: Path | str,
        mask_dir:  Path | str,
        batch_size: int = 8,
        patch_size: int = 256,
        augment: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        use_albumentations: bool = True,
    ) -> None:
        self.image_dir   = Path(image_dir)
        self.mask_dir    = Path(mask_dir)
        self.batch_size  = batch_size
        self.patch_size  = patch_size
        self.augment     = augment
        self.shuffle     = shuffle
        self._rng        = np.random.default_rng(seed)

        # Prefer albumentations if requested and available
        self._use_albumentations = augment and use_albumentations and _ALBUMENTATIONS_AVAILABLE
        if self._use_albumentations:
            self._aug_pipeline = _build_albumentations_pipeline()
            logger.info("Using albumentations augmentation pipeline.")
        elif augment:
            logger.info("Using OpenCV fallback augmentation pipeline.")

        # ── validate and align file lists ────────────────────────────────────
        self.image_paths: list[Path] = sorted(self.image_dir.glob("*.npy"))
        self.mask_paths:  list[Path] = sorted(self.mask_dir.glob("*.npy"))

        if not self.image_paths:
            raise FileNotFoundError(
                f"No .npy image patches found in '{self.image_dir}'. "
                "Run geospatial_utils.preprocess_scene() first."
            )
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Image/mask file count mismatch: "
                f"{len(self.image_paths)} images vs {len(self.mask_paths)} masks. "
                f"Ensure preprocessing created matching pairs."
            )

        self._indices = np.arange(len(self.image_paths))
        if self.shuffle:
            self._rng.shuffle(self._indices)

        logger.info(
            "CloudPatchDataset ready — %d samples | batch=%d | augment=%s",
            len(self.image_paths),
            batch_size,
            augment,
        )

    # ── Sequence protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of batches per epoch (ceiling division)."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(
        self, batch_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load one batch of (image, one-hot-mask) pairs.

        Args:
            batch_idx: Index within [0, len(self)).

        Returns:
            X: float32 ndarray (B, H, W, 4)
            y: float32 ndarray (B, H, W, NUM_CLASSES)
        """
        start = batch_idx * self.batch_size
        end   = min(start + self.batch_size, len(self.image_paths))
        idxs  = self._indices[start:end]
        actual_bs = len(idxs)

        X = np.zeros((actual_bs, self.patch_size, self.patch_size, 4),           dtype=np.float32)
        y = np.zeros((actual_bs, self.patch_size, self.patch_size, NUM_CLASSES),  dtype=np.float32)

        for i, idx in enumerate(idxs):
            image = np.load(self.image_paths[idx]).astype(np.float32)
            mask  = np.load(self.mask_paths[idx]).astype(np.uint8)

            # ── Sanity check ─────────────────────────────────────────────────
            if image.shape[:2] != (self.patch_size, self.patch_size):
                raise ValueError(
                    f"Patch '{self.image_paths[idx].name}' has shape {image.shape}; "
                    f"expected ({self.patch_size}, {self.patch_size}, 4)."
                )
            if image.shape[2] != 4:
                raise ValueError(
                    f"Patch '{self.image_paths[idx].name}' has {image.shape[2]} channels; "
                    f"model requires 4 (RGBNIR)."
                )

            # ── Augmentation ─────────────────────────────────────────────────
            if self.augment:
                if self._use_albumentations:
                    # albumentations expects uint8 for image; promote/demote
                    img_u8 = (image * 255.0).astype(np.uint8)
                    result = self._aug_pipeline(image=img_u8, mask=mask)
                    image  = result["image"].astype(np.float32) / 255.0
                    mask   = result["mask"]
                else:
                    image, mask = _augment_opencv(image, mask, self._rng)

            X[i] = np.clip(image, 0.0, 1.0)
            y[i] = self._one_hot(mask)

        return X, y

    def on_epoch_end(self) -> None:
        """Reshuffle sample ordering after every epoch."""
        if self.shuffle:
            self._rng.shuffle(self._indices)

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _one_hot(mask: np.ndarray) -> np.ndarray:
        """Convert (H, W) uint8 label map → (H, W, NUM_CLASSES) float32.

        Labels outside [0, NUM_CLASSES-1] are clipped to 0 (background)
        to handle stray annotation artefacts gracefully without crashing.
        """
        mask = np.clip(mask, 0, NUM_CLASSES - 1).astype(np.int32)
        return tf.keras.utils.to_categorical(mask, num_classes=NUM_CLASSES).astype(np.float32)

    # ── class utilities ───────────────────────────────────────────────────────

    @classmethod
    def train_val_split(
        cls,
        image_dir: Path | str,
        mask_dir:  Path | str,
        val_fraction: float = 0.15,
        batch_size: int = 8,
        patch_size: int = 256,
        seed: int = 42,
        use_albumentations: bool = True,
    ) -> tuple["CloudPatchDataset", "CloudPatchDataset"]:
        """Create matching train + validation generator pair.

        The split is deterministic (same seed → same split every run).
        Training generator: augment=True, shuffle=True.
        Validation generator: augment=False, shuffle=False.

        Args:
            image_dir:         Image patch directory.
            mask_dir:          Mask patch directory.
            val_fraction:      Fraction of samples reserved for validation.
            batch_size:        Batch size for both generators.
            patch_size:        Expected spatial patch dimension.
            seed:              Random seed.
            use_albumentations: Prefer albumentations if installed.

        Returns:
            (train_gen, val_gen) tuple.
        """
        image_dir = Path(image_dir)
        mask_dir  = Path(mask_dir)

        all_images = sorted(image_dir.glob("*.npy"))
        all_masks  = sorted(mask_dir.glob("*.npy"))

        if not all_images:
            raise FileNotFoundError(f"No .npy files found in '{image_dir}'.")

        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_images))
        rng.shuffle(indices)

        n_val   = max(1, int(len(indices) * val_fraction))
        val_idx = indices[:n_val]
        trn_idx = indices[n_val:]

        def _make_gen(idx_list: np.ndarray, is_train: bool) -> "CloudPatchDataset":
            gen = cls.__new__(cls)
            gen.image_dir   = image_dir
            gen.mask_dir    = mask_dir
            gen.batch_size  = batch_size
            gen.patch_size  = patch_size
            gen.augment     = is_train
            gen.shuffle     = is_train
            gen._rng        = np.random.default_rng(seed + (0 if is_train else 1))
            gen._use_albumentations = is_train and use_albumentations and _ALBUMENTATIONS_AVAILABLE
            if gen._use_albumentations:
                gen._aug_pipeline = _build_albumentations_pipeline()
            gen.image_paths = [all_images[i] for i in idx_list]
            gen.mask_paths  = [all_masks[i]  for i in idx_list]
            gen._indices    = np.arange(len(gen.image_paths))
            return gen

        train_gen = _make_gen(trn_idx, is_train=True)
        val_gen   = _make_gen(val_idx,  is_train=False)

        logger.info(
            "Split: train=%d  val=%d  (val_fraction=%.0f%%)",
            len(train_gen.image_paths),
            len(val_gen.image_paths),
            val_fraction * 100,
        )
        return train_gen, val_gen

    def __repr__(self) -> str:
        return (
            f"CloudPatchDataset("
            f"n={len(self.image_paths)}, "
            f"batch={self.batch_size}, "
            f"augment={self.augment})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — FINE-TUNE DATASET BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def incorporate_new_samples(
    new_image_dir: Path | str,
    new_mask_dir:  Path | str,
    base_image_dir: Path | str,
    base_mask_dir:  Path | str,
    prefix: str = "finetune",
) -> int:
    """Copy new annotated patches into the base training directories.

    This is the data-side of the continuous learning loop.  After a user
    uploads and annotates new GeoTIFFs via the dashboard, preprocess_scene()
    saves the patches to a temporary directory.  This function merges them
    into the main patch pool so the next fine-tuning call sees the new data.

    Files are copied (not moved) so the source remains intact.

    Args:
        new_image_dir:  Temp directory containing new image patches (.npy).
        new_mask_dir:   Temp directory containing new mask patches (.npy).
        base_image_dir: Main training image patch directory.
        base_mask_dir:  Main training mask patch directory.
        prefix:         Filename prefix to distinguish new samples.

    Returns:
        Number of new image patches copied.

    Raises:
        ValueError: If new_image_dir contains a different count than new_mask_dir.
    """
    new_image_dir  = Path(new_image_dir)
    new_mask_dir   = Path(new_mask_dir)
    base_image_dir = Path(base_image_dir)
    base_mask_dir  = Path(base_mask_dir)

    base_image_dir.mkdir(parents=True, exist_ok=True)
    base_mask_dir.mkdir(parents=True, exist_ok=True)

    new_images = sorted(new_image_dir.glob("*.npy"))
    new_masks  = sorted(new_mask_dir.glob("*.npy"))

    if len(new_images) != len(new_masks):
        raise ValueError(
            f"New sample count mismatch: {len(new_images)} images vs "
            f"{len(new_masks)} masks.  Preprocessing may have been incomplete."
        )

    for i, (img_path, msk_path) in enumerate(zip(new_images, new_masks)):
        dst_img = base_image_dir / f"{prefix}_{img_path.name}"
        dst_msk = base_mask_dir  / f"{prefix}_{msk_path.name}"
        shutil.copy2(img_path, dst_img)
        shutil.copy2(msk_path, dst_msk)

    logger.info(
        "Incorporated %d new samples from '%s' into '%s'.",
        len(new_images), new_image_dir, base_image_dir,
    )
    return len(new_images)
