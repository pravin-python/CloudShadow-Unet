"""
model.py — Deep Learning Architecture & Optimization
=====================================================
Principal Engineer: CloudShadow-UNet Project
─────────────────────────────────────────────────────────────────────────────
Implements the complete deep learning stack:

  • Multi-Class U-Net (4-channel RGBNIR input, 3-class softmax output)
  • Mixed-Precision Training  (bfloat16 on TPU/Ampere, float16 on older GPU)
  • Multi-Class Dice Loss     (combats severe class imbalance)
  • Combined Dice + CCE Loss  (stable gradients in early epochs)
  • Epoch-accumulating Dice Coefficient metric  (unbiased over full epoch)
  • Epoch-accumulating Mean IoU metric          (Jaccard Index, hard labels)

Architecture Numbers (default config):
  Encoder : 64 → 128 → 256 → 512 filters  (4 stages, MaxPool2×2)
  Bottleneck : 1024 filters
  Decoder : 512 → 256 → 128 → 64 filters  (4 stages, Conv2DTranspose)
  Output : Conv2D(1×1) → softmax → (B, H, W, 3)
  Parameters : ≈31 M  (fits single 8-GB GPU at batch=8, patch=256)

Mixed Precision Notes:
  • bfloat16  → preferred on Google TPU v4 and NVIDIA Ampere (A100, RTX 30xx)
  • float16   → use on Turing / Volta / older Ampere with tensor cores
  • float32   → fallback on CPU or when precision debugging is needed
  • The final softmax output layer is always cast to float32 to avoid
    numerical instability in the loss function regardless of global policy.
"""

from __future__ import annotations

import logging
from typing import Literal

import tensorflow as tf
from tensorflow.keras import Model, layers

logger = logging.getLogger(__name__)

# ─── precision constant ───────────────────────────────────────────────────────
_SMOOTH: float = 1e-6          # Label-smoothing / div-by-zero guard in losses
_NUM_CLASSES_DEFAULT: int = 3  # Background=0, Cloud=1, Shadow=2


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — MIXED PRECISION SETUP
# ═══════════════════════════════════════════════════════════════════════════════

PrecisionPolicy = Literal["bfloat16", "float16", "float32", "auto"]


def configure_precision(policy: PrecisionPolicy = "auto") -> str:
    """Configure TensorFlow mixed-precision policy and return active policy name.

    Calling this function before building the model is mandatory for mixed
    precision to take effect; it sets the global Keras dtype policy that
    every subsequent layer inherits.

    Policy selection logic ("auto"):
      1. NVIDIA Ampere (compute capability ≥ 8.0) → bfloat16
      2. NVIDIA Turing / Volta  (capability ≥ 7.0) → float16
      3. No GPU detected → float32  (CPU training)

    Args:
        policy: Explicit policy override, or "auto" for hardware detection.

    Returns:
        The name of the active policy that was set.

    Raises:
        ValueError: If an unsupported policy name is passed.
    """
    supported = {"bfloat16", "float16", "float32", "auto"}
    if policy not in supported:
        raise ValueError(f"policy must be one of {supported}; got '{policy}'")

    if policy == "auto":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            policy = "float32"
            logger.info("No GPU detected — using float32 precision.")
        else:
            # Query compute capability of the first GPU
            try:
                details = tf.config.experimental.get_device_details(gpus[0])
                cc = details.get("compute_capability", (0, 0))
                major = cc[0] if isinstance(cc, tuple) else 0
                if major >= 8:
                    policy = "bfloat16"   # Ampere: native bfloat16 tensor cores
                elif major >= 7:
                    policy = "float16"    # Turing / Volta: FP16 tensor cores
                else:
                    policy = "float32"    # Older GPUs: no tensor cores
            except Exception:
                policy = "float16"        # Safe default if detection fails

    tf.keras.mixed_precision.set_global_policy(policy)
    logger.info("Keras mixed precision policy set to: %s", policy)
    return policy


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — U-NET BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════

def _conv_relu_block(
    x: tf.Tensor,
    filters: int,
    dropout: float,
    name: str,
) -> tf.Tensor:
    """Two consecutive Conv2D(3×3, 'same', ReLU) + SpatialDropout2D.

    Design notes:
      • He-normal kernel init: optimal for ReLU activations (avoids vanishing
        gradient in deep nets — He et al., 2015).
      • No BatchNormalization: BN statistics are unreliable with the small
        effective batch sizes that come from large (384+) patch sizes and
        limited VRAM.  Dropout is the primary regulariser here.
      • SpatialDropout2D drops entire feature map channels, not individual
        pixels.  Adjacent pixels are highly correlated so pixel-level dropout
        provides almost no regularisation signal (Tompson et al., 2015).
      • The layer dtype inherits the global Keras policy, so intermediate
        tensors are automatically in float16/bfloat16 during mixed precision.

    Args:
        x:       Input feature map.
        filters: Number of convolution filters.
        dropout: SpatialDropout2D rate (0.0 disables dropout).
        name:    Block name prefix used in TensorBoard / model.summary().

    Returns:
        Output feature map tensor.
    """
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name=f"{name}_conv1",
    )(x)
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name=f"{name}_conv2",
    )(x)
    if dropout > 0.0:
        x = layers.SpatialDropout2D(dropout, name=f"{name}_sdrop")(x)
    return x


def _encoder_block(
    x: tf.Tensor,
    filters: int,
    dropout: float,
    name: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    """One contracting step: conv_block → save skip → MaxPool2D(2×2).

    Returns:
        skip:   Pre-pooling feature map — later concatenated in decoder.
        pooled: Downsampled feature map — fed to the next encoder stage.
    """
    skip = _conv_relu_block(x, filters, dropout=dropout, name=name)
    pooled = layers.MaxPooling2D(pool_size=2, strides=2, name=f"{name}_pool")(skip)
    return skip, pooled


def _decoder_block(
    x: tf.Tensor,
    skip: tf.Tensor,
    filters: int,
    dropout: float,
    name: str,
) -> tf.Tensor:
    """One expansive step: Conv2DTranspose → align → Concat(skip) → conv_block.

    Why Conv2DTranspose instead of Bilinear + Conv2D?
      Learnable upsampling kernel allows the decoder to develop cloud-edge
      specific upsampling behaviour, rather than generic bilinear smoothing.

    Spatial alignment guard:
      Odd-sized inputs (e.g., 513×513) can cause ±1 pixel mismatches between
      transposed output and the skip tensor.  The Resizing layer corrects this
      without discarding any feature data.

    Args:
        x:       Input from previous decoder stage or bottleneck.
        skip:    Skip-connection tensor from the matching encoder stage.
        filters: Number of conv filters after concatenation.
        dropout: Dropout rate.
        name:    Block name prefix.

    Returns:
        Output tensor after upsample → concat → conv.
    """
    x = layers.Conv2DTranspose(
        filters,
        kernel_size=2,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name}_up",
    )(x)

    # Guard: align spatial dims in case of ±1 pixel mismatch
    skip_h = tf.shape(skip)[1]
    skip_w = tf.shape(skip)[2]
    x = tf.image.resize(x, [skip_h, skip_w], method="bilinear", name=f"{name}_align")

    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = _conv_relu_block(x, filters, dropout=dropout, name=name)
    return x


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — U-NET MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def build_unet(
    input_shape: tuple[int, int, int] = (256, 256, 4),
    num_classes: int = _NUM_CLASSES_DEFAULT,
    base_filters: int = 64,
    depth: int = 4,
    dropout: float = 0.10,
    bottleneck_dropout: float = 0.30,
) -> Model:
    """Build and return the CloudShadow U-Net Keras model.

    The model is NOT compiled here — compilation is the responsibility of
    the training module.  Separating architecture from optimiser config makes
    it trivial to re-compile with a different LR for fine-tuning.

    Args:
        input_shape:        (H, W, C) — default (256, 256, 4) for RGBNIR.
        num_classes:        Segmentation output channels (default 3).
        base_filters:       Filters in first encoder stage; doubles each level.
        depth:              Number of encoder/decoder stages.
        dropout:            Dropout rate in standard conv blocks.
        bottleneck_dropout: Dropout rate in the bottleneck (higher = more reg).

    Returns:
        Uncompiled Keras functional Model.
    """
    inputs = layers.Input(shape=input_shape, name="rgbnir_input", dtype="float32")

    # ── Encoder path ─────────────────────────────────────────────────────────
    skips: list[tf.Tensor] = []
    x = inputs
    for level in range(depth):
        f = base_filters * (2 ** level)   # 64, 128, 256, 512
        skip, x = _encoder_block(x, filters=f, dropout=dropout, name=f"enc{level + 1}")
        skips.append(skip)

    # ── Bottleneck ────────────────────────────────────────────────────────────
    f_bn = base_filters * (2 ** depth)    # 1024
    x = _conv_relu_block(x, filters=f_bn, dropout=bottleneck_dropout, name="bottleneck")

    # ── Decoder path ─────────────────────────────────────────────────────────
    for level in reversed(range(depth)):
        f = base_filters * (2 ** level)   # 512, 256, 128, 64
        x = _decoder_block(x, skips[level], filters=f, dropout=dropout, name=f"dec{level + 1}")

    # ── Output head ───────────────────────────────────────────────────────────
    # dtype="float32" forces float32 output even when global policy is float16
    # or bfloat16.  Softmax must be numerically stable — never compute it in
    # reduced precision as this causes NaN loss on certain batches.
    outputs = layers.Conv2D(
        num_classes,
        kernel_size=1,
        padding="same",
        activation="softmax",
        kernel_initializer="glorot_uniform",
        dtype="float32",          # ← Critical: always float32 for softmax
        name="segmentation_map",
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="CloudShadow_UNet")
    logger.info(
        "U-Net built — input=%s  output=%s  params=%s",
        input_shape,
        outputs.shape,
        f"{model.count_params():,}",
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — CUSTOM LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class MultiClassDiceLoss(tf.keras.losses.Loss):
    """Macro-averaged Multi-Class Dice Loss for imbalanced segmentation.

    Mathematical definition per class c:
        Dice_c = (2 × Σ(y_true_c × y_pred_c) + ε)
                 ─────────────────────────────────────
                 (Σ y_true_c + Σ y_pred_c + ε)

    The final loss is:
        MultiDiceLoss = 1 − mean(Dice_c  for c in [0, num_classes))

    Why macro-average?
        Each class contributes equally to the gradient regardless of how many
        pixels it occupies.  This directly counteracts the background-dominated
        distribution of satellite cloud datasets.

    Both y_true and y_pred are expected in one-hot / softmax format:
        Shape: (batch, H, W, num_classes)

    Attributes:
        smooth: Additive smoothing to prevent 0/0 division.
    """

    def __init__(
        self,
        smooth: float = _SMOOTH,
        name: str = "multiclass_dice_loss",
    ) -> None:
        super().__init__(name=name)
        self.smooth = smooth

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Reduce over spatial + batch dims, keep class axis
        axes = [0, 1, 2]
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)          # (C,)
        sum_true     = tf.reduce_sum(y_true,           axis=axes)          # (C,)
        sum_pred     = tf.reduce_sum(y_pred,           axis=axes)          # (C,)

        dice_per_class = (2.0 * intersection + self.smooth) / (
            sum_true + sum_pred + self.smooth
        )
        return 1.0 - tf.reduce_mean(dice_per_class)

    def get_config(self) -> dict:
        return {**super().get_config(), "smooth": self.smooth}


class CombinedDiceCELoss(tf.keras.losses.Loss):
    """Weighted combination of Multi-Class Dice Loss + Categorical Cross-Entropy.

    Rationale:
        Dice Loss alone can produce very flat / zero gradients in the first
        few epochs when predictions are near-uniform (∇Dice ≈ 0 at Dice≈0.5).
        A 30 % CCE contribution provides strong per-pixel gradients during
        initialisation, which then fade as Dice Loss takes over.

        Combined = α × DiceLoss + (1 − α) × CategoricalCrossEntropy
        Default α = 0.70

    Attributes:
        alpha:  Dice Loss weight [0, 1].  CCE weight = 1 − alpha.
        smooth: Smoothing constant for the Dice component.
    """

    def __init__(
        self,
        alpha: float = 0.70,
        smooth: float = _SMOOTH,
        name: str = "combined_dice_ce",
    ) -> None:
        super().__init__(name=name)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
        self.alpha = alpha
        self.smooth = smooth
        self._dice = MultiClassDiceLoss(smooth=smooth)
        self._cce  = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        dice = self._dice(y_true, y_pred)
        cce  = self._cce(y_true, y_pred)
        return self.alpha * dice + (1.0 - self.alpha) * cce

    def get_config(self) -> dict:
        return {**super().get_config(), "alpha": self.alpha, "smooth": self.smooth}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — CUSTOM METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class DiceCoefficient(tf.keras.metrics.Metric):
    """Epoch-level macro-averaged Dice Coefficient (higher = better).

    Accumulates true intersections and sums across all batches in the epoch,
    then computes a single epoch-level Dice score.  This avoids the downward
    bias of averaging per-batch Dice scores when batches have varying class
    frequencies (which they always do with a random sampler).

    Shape contract:
        y_true: (batch, H, W, num_classes)  float32 one-hot
        y_pred: (batch, H, W, num_classes)  float32 softmax probabilities
    """

    def __init__(
        self,
        num_classes: int = _NUM_CLASSES_DEFAULT,
        smooth: float = _SMOOTH,
        name: str = "dice_coeff",
    ) -> None:
        super().__init__(name=name)
        self.num_classes = num_classes
        self.smooth = smooth
        self._inter = self.add_weight(name="inter", shape=(num_classes,), initializer="zeros")
        self._denom = self.add_weight(name="denom", shape=(num_classes,), initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight=None,
    ) -> None:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        axes = [0, 1, 2]
        self._inter.assign_add(tf.reduce_sum(y_true * y_pred, axis=axes))
        self._denom.assign_add(tf.reduce_sum(y_true + y_pred, axis=axes))

    def result(self) -> tf.Tensor:
        dice_per_class = (2.0 * self._inter + self.smooth) / (self._denom + self.smooth)
        return tf.reduce_mean(dice_per_class)

    def reset_state(self) -> None:
        self._inter.assign(tf.zeros(self.num_classes))
        self._denom.assign(tf.zeros(self.num_classes))

    def get_config(self) -> dict:
        return {**super().get_config(), "num_classes": self.num_classes, "smooth": self.smooth}


class MeanIoU(tf.keras.metrics.Metric):
    """Epoch-level macro-averaged Mean Intersection-over-Union (Jaccard Index).

    Unlike the built-in tf.keras.metrics.MeanIoU, this version:
      • Works directly with softmax probabilities via argmax → one-hot.
      • Accumulates over the full epoch before computing the score, eliminating
        per-batch estimator bias.
      • Uses additive smoothing to handle classes absent from a batch.

    Shape contract:
        y_true: (batch, H, W, num_classes)  float32 one-hot
        y_pred: (batch, H, W, num_classes)  float32 softmax probabilities
    """

    def __init__(
        self,
        num_classes: int = _NUM_CLASSES_DEFAULT,
        smooth: float = _SMOOTH,
        name: str = "mean_iou",
    ) -> None:
        super().__init__(name=name)
        self.num_classes = num_classes
        self.smooth = smooth
        self._inter = self.add_weight(name="inter", shape=(num_classes,), initializer="zeros")
        self._union = self.add_weight(name="union", shape=(num_classes,), initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight=None,
    ) -> None:
        y_true = tf.cast(y_true, tf.float32)
        # Hard argmax predictions for IoU (IoU is always computed on discrete labels)
        pred_hard = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=self.num_classes)
        pred_hard = tf.cast(pred_hard, tf.float32)

        axes = [0, 1, 2]
        inter = tf.reduce_sum(y_true * pred_hard, axis=axes)
        union = tf.reduce_sum(y_true + pred_hard, axis=axes) - inter

        self._inter.assign_add(inter)
        self._union.assign_add(union)

    def result(self) -> tf.Tensor:
        iou_per_class = (self._inter + self.smooth) / (self._union + self.smooth)
        return tf.reduce_mean(iou_per_class)

    def reset_state(self) -> None:
        self._inter.assign(tf.zeros(self.num_classes))
        self._union.assign(tf.zeros(self.num_classes))

    def get_config(self) -> dict:
        return {**super().get_config(), "num_classes": self.num_classes, "smooth": self.smooth}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — CUSTOM OBJECTS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_OBJECTS: dict = {
    "MultiClassDiceLoss": MultiClassDiceLoss,
    "CombinedDiceCELoss": CombinedDiceCELoss,
    "DiceCoefficient":    DiceCoefficient,
    "MeanIoU":            MeanIoU,
}
"""Pass this dict to tf.keras.models.load_model(custom_objects=CUSTOM_OBJECTS)
so Keras can deserialise the saved loss and metric classes."""


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — COMPILED MODEL FACTORY (convenience)
# ═══════════════════════════════════════════════════════════════════════════════

def build_and_compile(
    input_shape: tuple[int, int, int] = (256, 256, 4),
    num_classes: int = _NUM_CLASSES_DEFAULT,
    base_filters: int = 64,
    depth: int = 4,
    dropout: float = 0.10,
    bottleneck_dropout: float = 0.30,
    learning_rate: float = 1e-4,
    dice_alpha: float = 0.70,
    precision: PrecisionPolicy = "auto",
) -> Model:
    """Build, configure precision, compile, and return a ready-to-train model.

    This is the one-stop convenience factory used by train.py.

    Args:
        input_shape:        (H, W, C) — default (256, 256, 4).
        num_classes:        Segmentation classes.
        base_filters:       First encoder filter count.
        depth:              Encoder/decoder stages.
        dropout:            Conv block dropout.
        bottleneck_dropout: Bottleneck dropout (higher to regularise).
        learning_rate:      Initial Adam learning rate.
        dice_alpha:         Dice weight in combined loss (rest → CCE).
        precision:          Mixed precision policy — see configure_precision().

    Returns:
        Compiled Keras Model.
    """
    configure_precision(precision)

    model = build_unet(
        input_shape=input_shape,
        num_classes=num_classes,
        base_filters=base_filters,
        depth=depth,
        dropout=dropout,
        bottleneck_dropout=bottleneck_dropout,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            # Loss scale manager is injected automatically by Keras when the
            # policy is float16; bfloat16 does not need it (wider range).
        ),
        loss=CombinedDiceCELoss(alpha=dice_alpha),
        metrics=[
            DiceCoefficient(num_classes=num_classes),
            MeanIoU(num_classes=num_classes),
        ],
    )

    logger.info(
        "Model compiled — LR=%.2e  dice_alpha=%.2f  precision=%s",
        learning_rate, dice_alpha, tf.keras.mixed_precision.global_policy().name,
    )
    return model


# ─── CLI quick-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = build_and_compile(precision="float32")
    m.summary(line_length=110)
