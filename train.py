"""
train.py — Model Training & Continuous Fine-Tuning
===================================================
Principal Engineer: CloudShadow-UNet Project
─────────────────────────────────────────────────────────────────────────────
Implements the complete training pipeline with two primary modes:

  1. INITIAL TRAINING  — Train a fresh model from scratch on the full patch
       dataset.  Uses the compiled model from model.py and the lazy-loading
       generator from dataset.py.

  2. FINE-TUNING / CONTINUOUS LEARNING — Load an existing checkpoint and
       resume training on a combination of existing + newly uploaded
       annotated samples.  Learning rate is automatically reduced 10× to
       prevent overwriting previously learned representations.

Both modes share the same callback stack:
  • ModelCheckpoint       — saves best epoch by val_loss (not val_dice, since
                            val_loss has finer gradient signal via CCE term)
  • ReduceLROnPlateau     — halves LR when val_loss stalls for N epochs
  • EarlyStopping         — aborts training if val_loss doesn't improve
  • TensorBoard           — loss curves, weight histograms, LR tracking
  • CSVLogger             — flat CSV epoch log for post-hoc analysis
  • LearningRateLogger    — custom callback logging exact LR each epoch

Usage (CLI):
  # Fresh training
  python train.py --mode train --config configs/unet_baseline.yaml

  # Fine-tune existing weights with new patches
  python train.py --mode finetune \
      --checkpoint models/best_weights.h5 \
      --new_images /tmp/new_patches/images \
      --new_masks  /tmp/new_patches/masks
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — TRAINING CONFIGURATION DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Typed, validated container for all training hyperparameters.

    Populated either from a YAML file (CLI) or directly from Python
    (Streamlit dashboard fine-tuning call).

    Attributes:
        image_dir:           Directory of preprocessed image patches.
        mask_dir:            Directory of preprocessed mask patches.
        model_dir:           Directory where checkpoint + final model are saved.
        log_dir:             TensorBoard / CSV log directory.
        patch_size:          Spatial size of each training tile.
        batch_size:          Samples per training batch.
        epochs:              Maximum training epochs.
        val_fraction:        Held-out validation fraction.
        learning_rate:       Initial Adam learning rate.
        dice_alpha:          Dice Loss weight in combined loss.
        base_filters:        U-Net first-stage filter count.
        depth:               Encoder/decoder depth.
        dropout:             Conv block dropout rate.
        bottleneck_dropout:  Bottleneck dropout rate.
        reduce_lr_patience:  ReduceLROnPlateau patience (epochs).
        early_stop_patience: EarlyStopping patience (epochs).
        precision:           Mixed precision policy.
        seed:                Global random seed.
        workers:             Keras generator worker threads.
        use_albumentations:  Prefer albumentations augmentation.
        resume_checkpoint:   Path to weights file to resume from (optional).
    """

    image_dir: Path = field(default_factory=lambda: Path("data/patches"))
    mask_dir:  Path = field(default_factory=lambda: Path("data/masks"))
    model_dir: Path = field(default_factory=lambda: Path("models"))
    log_dir:   Path = field(default_factory=lambda: Path("logs"))

    patch_size:   int   = 256
    batch_size:   int   = 8
    epochs:       int   = 100
    val_fraction: float = 0.15
    seed:         int   = 42

    learning_rate:      float = 1e-4
    dice_alpha:         float = 0.70
    base_filters:       int   = 64
    depth:              int   = 4
    dropout:            float = 0.10
    bottleneck_dropout: float = 0.30

    reduce_lr_patience:  int = 5
    early_stop_patience: int = 15

    precision:           str  = "auto"
    workers:             int  = 4
    use_albumentations:  bool = True
    resume_checkpoint:   Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        """Load config from a YAML file, with environment variable overrides.

        Environment variable priority:
            DATA_DIR   → image_dir parent
            MODEL_PATH → resume_checkpoint
            PATCH_SIZE → patch_size
        """
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        data_dir = Path(os.environ.get("DATA_DIR", "data"))
        cfg = cls(
            image_dir   = data_dir / "patches",
            mask_dir    = data_dir / "masks",
            model_dir   = Path(os.environ.get("MODEL_PATH", raw.get("model_dir", "models"))),
            log_dir     = Path(raw.get("log_dir", "logs")),
            patch_size  = int(os.environ.get("PATCH_SIZE", raw.get("patch_size", 256))),
            batch_size  = int(raw.get("batch_size", 8)),
            epochs      = int(raw.get("epochs", 100)),
            val_fraction= float(raw.get("val_fraction", 0.15)),
            seed        = int(raw.get("seed", 42)),
            learning_rate       = float(raw.get("learning_rate", 1e-4)),
            dice_alpha          = float(raw.get("dice_alpha", 0.70)),
            base_filters        = int(raw.get("base_filters", 64)),
            depth               = int(raw.get("depth", 4)),
            dropout             = float(raw.get("dropout_rate", 0.10)),
            bottleneck_dropout  = float(raw.get("bottleneck_dropout", 0.30)),
            reduce_lr_patience  = int(raw.get("reduce_lr_patience", 5)),
            early_stop_patience = int(raw.get("early_stop_patience", 15)),
            precision           = raw.get("precision", "auto"),
            workers             = int(raw.get("workers", 4)),
            use_albumentations  = bool(raw.get("use_albumentations", True)),
        )
        resume = os.environ.get("RESUME_CHECKPOINT", raw.get("resume_checkpoint"))
        if resume:
            cfg.resume_checkpoint = Path(resume)
        return cfg


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — CUSTOM CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

class LearningRateLogger(tf.keras.callbacks.Callback):
    """Logs the exact learning rate at the end of each epoch.

    ReduceLROnPlateau modifies the LR silently by default.  This callback
    makes the current LR visible in TensorBoard and the CSV log so you can
    correlate LR drops with val_loss improvements.
    """

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        import tensorflow as tf
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        if logs is not None:
            logs["lr"] = lr


class EpochTimingCallback(tf.keras.callbacks.Callback):
    """Logs wall-clock time per epoch — useful for estimating total run time."""

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self._t0 = time.time()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        elapsed = time.time() - self._t0
        logger.info("Epoch %d completed in %.1f s", epoch + 1, elapsed)
        if logs is not None:
            logs["epoch_time_s"] = round(elapsed, 1)


def build_callbacks(cfg: TrainingConfig, run_tag: str = "run") -> list:
    """Construct and return the full callback stack for model.fit().

    Args:
        cfg:     TrainingConfig with path and patience settings.
        run_tag: String appended to log/checkpoint filenames to distinguish
                 runs (e.g., "initial", "finetune_20260412").

    Returns:
        List of initialised Keras Callback objects.
    """
    import tensorflow as tf

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = cfg.model_dir / f"best_{run_tag}.h5"
    csv_path  = cfg.log_dir   / f"training_{run_tag}.csv"
    tb_path   = cfg.log_dir   / f"tensorboard_{run_tag}"

    callbacks = [
        # ── 1. Save best checkpoint ──────────────────────────────────────────
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,  # Smaller files; load with load_weights()
            verbose=1,
        ),

        # ── 2. Reduce LR on plateau ──────────────────────────────────────────
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,              # Halve the LR
            patience=cfg.reduce_lr_patience,
            min_lr=1e-7,             # Never go below 0.1 μ-rate
            verbose=1,
        ),

        # ── 3. Early stopping ────────────────────────────────────────────────
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stop_patience,
            restore_best_weights=True,   # Roll back to best checkpoint on stop
            verbose=1,
        ),

        # ── 4. TensorBoard ───────────────────────────────────────────────────
        tf.keras.callbacks.TensorBoard(
            log_dir=str(tb_path),
            histogram_freq=1,           # Weight histograms every epoch
            update_freq="epoch",
            profile_batch=0,            # Disable profiler to avoid overhead
        ),

        # ── 5. CSV logger ────────────────────────────────────────────────────
        tf.keras.callbacks.CSVLogger(
            str(csv_path),
            separator=",",
            append=True,               # append=True supports resume correctly
        ),

        # ── 6. LR + timing ───────────────────────────────────────────────────
        LearningRateLogger(),
        EpochTimingCallback(),
    ]

    logger.info("Checkpoint path : %s", ckpt_path)
    logger.info("TensorBoard path: %s  →  tensorboard --logdir %s", tb_path, cfg.log_dir)
    return callbacks


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — GPU INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def _init_gpus() -> None:
    """Enable memory growth on all detected GPUs.

    Without memory growth, TensorFlow pre-allocates the entire GPU VRAM at
    startup — which prevents other processes from using the same GPU and can
    cause OOM on systems where VRAM is shared.

    Must be called BEFORE any TensorFlow computation graph is built.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.warning("No GPU detected — training will use CPU (slow).")
        return

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info("Memory growth enabled on %d GPU(s).", len(gpus))


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — INITIAL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_from_scratch(cfg: TrainingConfig) -> dict:
    """Train a fresh U-Net model from random initialisation.

    Steps:
      1. Configure GPU memory growth
      2. Build + compile model with mixed precision
      3. Create train/val generators
      4. Attach callbacks
      5. Run model.fit()
      6. Save final model (SavedModel format for portability)
      7. Return epoch history dict

    Args:
        cfg: Fully populated TrainingConfig.

    Returns:
        Keras History.history dict mapping metric name → list of epoch values.
    """
    import tensorflow as tf
    from dataset import CloudPatchDataset
    from model import build_and_compile

    _init_gpus()

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_and_compile(
        input_shape        = (cfg.patch_size, cfg.patch_size, 4),
        num_classes        = 3,
        base_filters       = cfg.base_filters,
        depth              = cfg.depth,
        dropout            = cfg.dropout,
        bottleneck_dropout = cfg.bottleneck_dropout,
        learning_rate      = cfg.learning_rate,
        dice_alpha         = cfg.dice_alpha,
        precision          = cfg.precision,
    )
    model.summary(line_length=110)

    # Optional: load weights to resume from a previous run
    if cfg.resume_checkpoint and Path(cfg.resume_checkpoint).exists():
        model.load_weights(str(cfg.resume_checkpoint))
        logger.info("Loaded weights from checkpoint: %s", cfg.resume_checkpoint)

    # ── Data generators ───────────────────────────────────────────────────────
    train_gen, val_gen = CloudPatchDataset.train_val_split(
        image_dir         = cfg.image_dir,
        mask_dir          = cfg.mask_dir,
        val_fraction      = cfg.val_fraction,
        batch_size        = cfg.batch_size,
        patch_size        = cfg.patch_size,
        seed              = cfg.seed,
        use_albumentations= cfg.use_albumentations,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = build_callbacks(cfg, run_tag="initial")

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info(
        "Starting training: epochs=%d  batch=%d  train_samples=%d  val_samples=%d",
        cfg.epochs, cfg.batch_size,
        len(train_gen.image_paths), len(val_gen.image_paths),
    )

    history = model.fit(
        train_gen,
        validation_data  = val_gen,
        epochs           = cfg.epochs,
        callbacks        = callbacks,
        workers          = cfg.workers,
        use_multiprocessing = True,
        verbose          = 1,
    )

    # ── Persist final model ───────────────────────────────────────────────────
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    final_path = cfg.model_dir / "final_model.keras"
    model.save(str(final_path))
    logger.info("Final model saved → %s", final_path)

    _log_peak_metrics(history.history)
    return history.history


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — FINE-TUNING / CONTINUOUS LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

def fine_tune(
    checkpoint_path: Path,
    cfg: TrainingConfig,
    new_image_dir: Optional[Path] = None,
    new_mask_dir:  Optional[Path] = None,
    fine_tune_lr_scale: float = 0.10,
    fine_tune_epochs: int = 20,
    progress_callback=None,
) -> dict:
    """Resume training from an existing checkpoint with optional new data.

    Fine-tuning strategy:
      • Load the model architecture + weights from checkpoint.
      • Re-compile with LR scaled down by fine_tune_lr_scale (default 10×
        reduction) to avoid catastrophically overwriting learned features.
      • If new image/mask patches are provided, merge them into the base
        training directory first.
      • Run model.fit() for fine_tune_epochs (shorter than initial training).
      • Save updated best checkpoint with a timestamped tag.

    The 10× LR reduction is a standard fine-tuning heuristic:
      initial LR = 1e-4 → fine-tune LR = 1e-5
      This keeps the existing feature representations mostly intact while
      allowing the model to adapt to the style/domain of new samples.

    Args:
        checkpoint_path:    Path to the .h5 weights file.
        cfg:                TrainingConfig (paths, batch size, patch size …).
        new_image_dir:      Optional new image patches to incorporate.
        new_mask_dir:       Optional new mask patches to incorporate.
        fine_tune_lr_scale: LR multiplier relative to cfg.learning_rate.
        fine_tune_epochs:   Maximum epochs for this fine-tuning run.
        progress_callback:  Optional callable(epoch, total_epochs, logs) for
                            real-time progress reporting (used by dashboard).

    Returns:
        Keras History.history dict.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    import tensorflow as tf
    from dataset import CloudPatchDataset, incorporate_new_samples
    from model import CUSTOM_OBJECTS, CombinedDiceCELoss, DiceCoefficient, MeanIoU

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _init_gpus()

    # ── Incorporate new data ──────────────────────────────────────────────────
    if new_image_dir is not None and new_mask_dir is not None:
        n_new = incorporate_new_samples(
            new_image_dir  = new_image_dir,
            new_mask_dir   = new_mask_dir,
            base_image_dir = cfg.image_dir,
            base_mask_dir  = cfg.mask_dir,
            prefix         = f"finetune_{int(time.time())}",
        )
        logger.info("Incorporated %d new samples into training pool.", n_new)

    # ── Load model ────────────────────────────────────────────────────────────
    # Try loading as full SavedModel first; fall back to weights-only h5.
    try:
        model = tf.keras.models.load_model(
            str(checkpoint_path.parent / "final_model.keras"),
            custom_objects=CUSTOM_OBJECTS,
        )
        logger.info("Loaded full SavedModel from '%s'.", checkpoint_path.parent)
    except Exception:
        # Weights-only checkpoint: rebuild architecture then load weights
        from model import build_unet
        model = build_unet(
            input_shape        = (cfg.patch_size, cfg.patch_size, 4),
            num_classes        = 3,
            base_filters       = cfg.base_filters,
            depth              = cfg.depth,
            dropout            = cfg.dropout,
            bottleneck_dropout = cfg.bottleneck_dropout,
        )
        model.load_weights(str(checkpoint_path))
        logger.info("Loaded weights from '%s'.", checkpoint_path)

    # ── Re-compile with reduced LR ────────────────────────────────────────────
    ft_lr = cfg.learning_rate * fine_tune_lr_scale
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=ft_lr),
        loss      = CombinedDiceCELoss(alpha=cfg.dice_alpha),
        metrics   = [DiceCoefficient(num_classes=3), MeanIoU(num_classes=3)],
    )
    logger.info("Re-compiled model with fine-tune LR = %.2e", ft_lr)

    # ── Data generators ───────────────────────────────────────────────────────
    train_gen, val_gen = CloudPatchDataset.train_val_split(
        image_dir         = cfg.image_dir,
        mask_dir          = cfg.mask_dir,
        val_fraction      = cfg.val_fraction,
        batch_size        = cfg.batch_size,
        patch_size        = cfg.patch_size,
        seed              = cfg.seed,
        use_albumentations= cfg.use_albumentations,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    run_tag = f"finetune_{int(time.time())}"
    callbacks = build_callbacks(cfg, run_tag=run_tag)

    # Optional real-time progress reporting for the Streamlit dashboard
    if progress_callback is not None:
        callbacks.append(_DashboardProgressCallback(progress_callback, fine_tune_epochs))

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    logger.info(
        "Fine-tuning: epochs=%d  lr=%.2e  train=%d  val=%d",
        fine_tune_epochs, ft_lr,
        len(train_gen.image_paths), len(val_gen.image_paths),
    )

    history = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = fine_tune_epochs,
        callbacks       = callbacks,
        workers         = cfg.workers,
        use_multiprocessing = True,
        verbose         = 1,
    )

    # ── Save updated model ────────────────────────────────────────────────────
    updated_path = cfg.model_dir / f"finetuned_{run_tag}.keras"
    model.save(str(updated_path))
    logger.info("Fine-tuned model saved → %s", updated_path)

    _log_peak_metrics(history.history)
    return history.history


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — DASHBOARD PROGRESS CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

class _DashboardProgressCallback(tf.keras.callbacks.Callback):
    """Bridge between Keras training loop and a Streamlit progress reporter.

    Calls progress_fn(epoch, total, logs) at the end of each epoch.
    The dashboard can use this to update st.progress() bars without polling.

    Args:
        progress_fn: Callable(epoch: int, total: int, logs: dict) → None.
        total_epochs: Total number of fine-tuning epochs.
    """

    def __init__(self, progress_fn, total_epochs: int) -> None:
        super().__init__()
        self._fn = progress_fn
        self._total = total_epochs

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        try:
            self._fn(epoch + 1, self._total, logs or {})
        except Exception as e:
            logger.debug("Dashboard progress callback error (non-fatal): %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — LOGGING UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def _log_peak_metrics(history: dict) -> None:
    """Log the best validation metrics from a completed training run."""
    peaks = {
        "Best val_loss":       ("val_loss",      min),
        "Best val_dice_coeff": ("val_dice_coeff", max),
        "Best val_mean_iou":   ("val_mean_iou",   max),
    }
    for label, (key, fn) in peaks.items():
        values = history.get(key, [])
        if values:
            logger.info("%-25s : %.4f", label, fn(values))


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CloudShadow-UNet training & fine-tuning CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "finetune"],
        default="train",
        help="'train' = fresh training; 'finetune' = resume + new data",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/unet_baseline.yaml"),
        help="Path to YAML training configuration",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="[finetune mode] Path to .h5 weights checkpoint",
    )
    parser.add_argument(
        "--new_images",
        type=Path,
        default=None,
        help="[finetune mode] Directory of new image patches to incorporate",
    )
    parser.add_argument(
        "--new_masks",
        type=Path,
        default=None,
        help="[finetune mode] Directory of new mask patches to incorporate",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=20,
        help="[finetune mode] Number of fine-tuning epochs",
    )
    parser.add_argument(
        "--lr_scale",
        type=float,
        default=0.10,
        help="[finetune mode] LR multiplier relative to config learning_rate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import tensorflow as tf  # noqa: F401  (needed for type checks in callbacks)

    args = _parse_args()
    cfg  = TrainingConfig.from_yaml(args.config)

    if args.mode == "train":
        train_from_scratch(cfg)

    elif args.mode == "finetune":
        ckpt = args.checkpoint or cfg.model_dir / "best_initial.h5"
        fine_tune(
            checkpoint_path    = ckpt,
            cfg                = cfg,
            new_image_dir      = args.new_images,
            new_mask_dir       = args.new_masks,
            fine_tune_lr_scale = args.lr_scale,
            fine_tune_epochs   = args.finetune_epochs,
        )
