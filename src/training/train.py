"""
Training Entry Point
====================
Orchestrates the full training pipeline:
    1. Load YAML config
    2. (Optionally) enable mixed precision for memory-constrained GPUs
    3. Build U-Net model
    4. Compile with Combined Dice+CCE Loss, Dice, and IoU metrics
    5. Create train/val data generators
    6. Attach callbacks: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,
       TensorBoard, CSVLogger
    7. Run model.fit()

Usage:
    python src/training/train.py --config configs/unet_baseline.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path to resolve 'src' module
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
 
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CloudShadow U-Net")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/unet_baseline.yaml"),
        help="Path to YAML training configuration",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_callbacks(cfg: dict, model_dir: Path) -> list:
    """Construct the Keras callback stack.

    Callbacks:
        ModelCheckpoint   — saves best weights by val_loss (not val_dice,
                            because Dice can plateau while val_loss still
                            encodes fine-grained progress via the CCE term).
        ReduceLROnPlateau — halves LR after patience epochs of no val_loss
                            improvement; prevents overshooting local minima.
        EarlyStopping     — aborts training if val_loss stalls for longer,
                            freeing GPU time for hyperparameter search.
        TensorBoard       — loss/metric curves, histogram of weights.
        CSVLogger         — plain-text epoch log for post-hoc analysis.

    Args:
        cfg:       Parsed YAML config dict.
        model_dir: Directory where checkpoint files are written.

    Returns:
        List of initialised Keras Callback objects.
    """
    import tensorflow as tf

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / "best_weights.h5"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=cfg.get("reduce_lr_patience", 5),
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.get("early_stop_patience", 15),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            update_freq="epoch",
        ),
        tf.keras.callbacks.CSVLogger(
            str(log_dir / "training_log.csv"),
            append=True,
        ),
    ]
    return callbacks


def train(config_path: Path) -> None:
    """Full training pipeline.

    Args:
        config_path: Path to the YAML configuration file.
    """
    cfg = _load_config(config_path)
    logger.info("Loaded config: %s", config_path)

    # ── Optional mixed precision ──────────────────────────────────────────────
    if cfg.get("mixed_precision", False):
        import tensorflow as tf
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision enabled (float16 compute / float32 params)")

    import tensorflow as tf
    from src.model.generator import CloudSegmentationGenerator
    from src.model.losses import CombinedLoss, DiceCoefficient, MeanIoU
    from src.model.unet import build_unet

    # ── GPU memory growth (prevents OOM on shared machines) ──────────────────
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # ── Data generators ───────────────────────────────────────────────────────
    image_dir = Path(os.environ.get("DATA_DIR", "data")) / "patches"
    mask_dir = Path(os.environ.get("DATA_DIR", "data")) / "masks"
    patch_size = int(os.environ.get("PATCH_SIZE", cfg.get("patch_size", 256)))
    batch_size = cfg.get("batch_size", 8)

    train_gen, val_gen = CloudSegmentationGenerator.train_val_split(
        image_dir=image_dir,
        mask_dir=mask_dir,
        val_fraction=cfg.get("val_fraction", 0.15),
        batch_size=batch_size,
        patch_size=patch_size,
        seed=cfg.get("seed", 42),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    num_classes = int(os.environ.get("NUM_CLASSES", cfg.get("num_classes", 3)))
    model = build_unet(
        input_shape=(patch_size, patch_size, 4),
        num_classes=num_classes,
        base_filters=cfg.get("base_filters", 64),
        depth=cfg.get("depth", 4),
        dropout_rate=cfg.get("dropout_rate", 0.1),
        bottleneck_dropout=cfg.get("bottleneck_dropout", 0.3),
    )

    # Resume from checkpoint if specified
    resume_checkpoint = cfg.get("resume_checkpoint")
    if resume_checkpoint and Path(resume_checkpoint).exists():
        model.load_weights(resume_checkpoint)
        logger.info("Resumed weights from %s", resume_checkpoint)

    # ── Compile ───────────────────────────────────────────────────────────────
    dice_alpha = cfg.get("dice_alpha", 0.7)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.get("learning_rate", 1e-4)),
        loss=CombinedLoss(alpha=dice_alpha),
        metrics=[
            DiceCoefficient(num_classes=num_classes),
            MeanIoU(num_classes=num_classes),
        ],
    )
    model.summary(line_length=100)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    model_dir = Path(os.environ.get("MODEL_PATH", "models"))
    callbacks = build_callbacks(cfg, model_dir)

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info(
        "Starting training — epochs=%d  batch=%d  patch=%d",
        cfg.get("epochs", 100), batch_size, patch_size,
    )
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.get("epochs", 100),
        callbacks=callbacks,
        workers=cfg.get("workers", 4),
        use_multiprocessing=cfg.get("use_multiprocessing", True),
        verbose=1,
    )

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = model_dir / "final_model.keras"
    model.save(str(final_path))
    logger.info("Final model saved → %s", final_path)

    # ── Log peak metrics ──────────────────────────────────────────────────────
    val_dice = history.history.get("val_dice_coeff", [])
    val_iou = history.history.get("val_mean_iou", [])
    if val_dice:
        logger.info("Best val Dice Coeff : %.4f", max(val_dice))
    if val_iou:
        logger.info("Best val Mean IoU   : %.4f", max(val_iou))


if __name__ == "__main__":
    args = _parse_args()
    train(args.config)
