"""
Module 2 — Custom TensorFlow Data Generator
============================================
Lazily streams 4-channel image patches and one-hot encoded mask patches
from disk to GPU memory, with on-the-fly geometric augmentation.

The generator uses tf.keras.utils.Sequence so Keras can prefetch batches
on the CPU while the GPU trains, maximising throughput.

Augmentations applied randomly per sample:
    - Horizontal flip    (50 % probability)
    - Vertical flip      (50 % probability)
    - 90° rotation       (25 % probability per quadrant)
    - Random brightness  (±10 %, image only)
    - Gaussian noise     (σ ≤ 0.01, image only)

Design choices:
    - Augmentations are purely geometric (+ mild photometric) so mask
      labels remain valid after the same spatial transform.
    - One-hot encoding is deferred to generator time to keep .npy files
      compact (uint8 labels vs float32 channels).
    - Shuffle is per-epoch so every batch composition changes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

NUM_CLASSES: int = 3  # 0=Background, 1=Cloud, 2=Shadow


# ─── augmentation helpers ─────────────────────────────────────────────────────

def _random_flip(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random horizontal and/or vertical flip."""
    if rng.random() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if rng.random() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return image, mask


def _random_rotate90(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a random 0°/90°/180°/270° rotation."""
    k = rng.integers(0, 4)  # number of 90° CCW rotations
    if k > 0:
        image = np.rot90(image, k=k, axes=(0, 1))
        mask = np.rot90(mask, k=k)
    return image, mask


def _random_brightness(
    image: np.ndarray, rng: np.random.Generator, max_delta: float = 0.10
) -> np.ndarray:
    """Add a small uniform brightness offset to the image (not the mask)."""
    delta = rng.uniform(-max_delta, max_delta)
    return np.clip(image + delta, 0.0, 1.0).astype(np.float32)


def _random_gaussian_noise(
    image: np.ndarray, rng: np.random.Generator, max_sigma: float = 0.01
) -> np.ndarray:
    """Add pixel-wise Gaussian noise to simulate sensor jitter."""
    sigma = rng.uniform(0.0, max_sigma)
    noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0).astype(np.float32)


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply all augmentations consistently to an (image, mask) pair.

    Geometric ops use the same RNG state for image and mask so they
    receive identical spatial transforms.

    Args:
        image: float32 (H, W, 4) patch.
        mask:  uint8  (H, W) label patch.
        rng:   NumPy Generator instance (seeded per worker for reproducibility).

    Returns:
        Augmented (image, mask) tuple.
    """
    image, mask = _random_flip(image, mask, rng)
    image, mask = _random_rotate90(image, mask, rng)
    image = _random_brightness(image, rng)
    image = _random_gaussian_noise(image, rng)
    return image, mask


# ─── data generator ───────────────────────────────────────────────────────────

class CloudSegmentationGenerator(tf.keras.utils.Sequence):
    """Keras-compatible generator for cloud/shadow segmentation.

    Yields batches of:
        X: float32 tensor (batch, H, W, 4)  — normalised RGBNIR patches
        y: float32 tensor (batch, H, W, 3)  — one-hot encoded labels

    Args:
        image_dir:   Directory containing image patch .npy files.
        mask_dir:    Directory containing mask patch .npy files.
        batch_size:  Number of samples per batch (default 8).
        patch_size:  Expected spatial dimension of each patch (default 256).
        augment:     Whether to apply random augmentations (default True).
        shuffle:     Whether to shuffle file order each epoch (default True).
        seed:        Random seed for reproducibility.
    """

    def __init__(
        self,
        image_dir: Path | str,
        mask_dir: Path | str,
        batch_size: int = 8,
        patch_size: int = 256,
        augment: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.augment = augment
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)

        self.image_paths: list[Path] = sorted(self.image_dir.glob("*.npy"))
        self.mask_paths: list[Path] = sorted(self.mask_dir.glob("*.npy"))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No .npy files found in {self.image_dir}")
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Image/mask count mismatch: "
                f"{len(self.image_paths)} images vs {len(self.mask_paths)} masks"
            )

        self._indices = np.arange(len(self.image_paths))
        if self.shuffle:
            self._rng.shuffle(self._indices)

        logger.info(
            "Generator initialised — %d samples, batch=%d, augment=%s",
            len(self.image_paths), batch_size, augment,
        )

    # ── Sequence protocol ────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(
        self, batch_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load, (optionally augment), and return one batch.

        Args:
            batch_idx: Index of the batch within the epoch.

        Returns:
            (X, y) where X is float32 (B, H, W, 4) and
                         y is float32 (B, H, W, num_classes).
        """
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self.image_paths))
        batch_indices = self._indices[start:end]

        batch_size = len(batch_indices)
        X = np.zeros(
            (batch_size, self.patch_size, self.patch_size, 4), dtype=np.float32
        )
        y = np.zeros(
            (batch_size, self.patch_size, self.patch_size, NUM_CLASSES),
            dtype=np.float32,
        )

        for i, idx in enumerate(batch_indices):
            image = np.load(self.image_paths[idx]).astype(np.float32)
            mask = np.load(self.mask_paths[idx]).astype(np.uint8)

            # Validate spatial dimensions
            if image.shape[:2] != (self.patch_size, self.patch_size):
                raise ValueError(
                    f"Patch {self.image_paths[idx].name} has shape "
                    f"{image.shape} — expected ({self.patch_size}, "
                    f"{self.patch_size}, 4)"
                )

            if self.augment:
                image, mask = augment_pair(image, mask, self._rng)

            X[i] = image
            y[i] = self._one_hot(mask)

        return X, y

    def on_epoch_end(self) -> None:
        """Reshuffle indices at the end of each epoch."""
        if self.shuffle:
            self._rng.shuffle(self._indices)

    # ── internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _one_hot(mask: np.ndarray) -> np.ndarray:
        """Convert a (H, W) integer label map to (H, W, NUM_CLASSES) float32.

        Labels outside [0, NUM_CLASSES-1] are silently mapped to background
        (class 0) to handle annotation artefacts gracefully.

        Args:
            mask: uint8 (H, W) array with values in {0, 1, 2}.

        Returns:
            float32 (H, W, NUM_CLASSES) one-hot tensor.
        """
        mask = np.clip(mask, 0, NUM_CLASSES - 1)
        return tf.keras.utils.to_categorical(
            mask, num_classes=NUM_CLASSES
        ).astype(np.float32)

    # ── convenience factory ───────────────────────────────────────────────────

    @classmethod
    def train_val_split(
        cls,
        image_dir: Path | str,
        mask_dir: Path | str,
        val_fraction: float = 0.15,
        batch_size: int = 8,
        patch_size: int = 256,
        seed: int = 42,
    ) -> tuple["CloudSegmentationGenerator", "CloudSegmentationGenerator"]:
        """Create a train + validation generator pair from a single directory.

        Files are split deterministically so the split is reproducible.

        Args:
            image_dir:    Directory with image .npy patches.
            mask_dir:     Directory with mask  .npy patches.
            val_fraction: Fraction of data reserved for validation.
            batch_size:   Batch size for both generators.
            patch_size:   Expected patch spatial dimension.
            seed:         RNG seed for the split shuffle.

        Returns:
            (train_gen, val_gen) tuple of CloudSegmentationGenerator instances.
        """
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)

        all_images = sorted(image_dir.glob("*.npy"))
        all_masks = sorted(mask_dir.glob("*.npy"))

        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_images))
        rng.shuffle(indices)

        n_val = max(1, int(len(indices) * val_fraction))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        # Write symlink-free sub-lists by creating temp generators
        # that override their path lists directly.
        train_gen = cls.__new__(cls)
        train_gen.image_dir = image_dir
        train_gen.mask_dir = mask_dir
        train_gen.batch_size = batch_size
        train_gen.patch_size = patch_size
        train_gen.augment = True
        train_gen.shuffle = True
        train_gen._rng = np.random.default_rng(seed)
        train_gen.image_paths = [all_images[i] for i in train_idx]
        train_gen.mask_paths = [all_masks[i] for i in train_idx]
        train_gen._indices = np.arange(len(train_gen.image_paths))

        val_gen = cls.__new__(cls)
        val_gen.image_dir = image_dir
        val_gen.mask_dir = mask_dir
        val_gen.batch_size = batch_size
        val_gen.patch_size = patch_size
        val_gen.augment = False
        val_gen.shuffle = False
        val_gen._rng = np.random.default_rng(seed)
        val_gen.image_paths = [all_images[i] for i in val_idx]
        val_gen.mask_paths = [all_masks[i] for i in val_idx]
        val_gen._indices = np.arange(len(val_gen.image_paths))

        logger.info(
            "Train/val split — train: %d  val: %d",
            len(train_gen.image_paths),
            len(val_gen.image_paths),
        )
        return train_gen, val_gen
