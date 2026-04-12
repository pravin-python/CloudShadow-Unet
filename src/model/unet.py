"""
Module 3 — Custom Multi-Class U-Net Architecture
=================================================
Implements the original Ronneberger et al. U-Net topology adapted for
4-band (RGBNIR) multi-spectral satellite imagery with 3-class softmax output.

Architecture summary
--------------------
Input  : (batch, H, W, 4)   — float32 RGBNIR patches
Output : (batch, H, W, 3)   — per-pixel softmax probability maps

Contracting path (encoder):
    4 × EncoderBlock → [Conv2D(3×3, ReLU) × 2 → Dropout → MaxPool2D(2×2)]
    Channels: 64 → 128 → 256 → 512

Bottleneck:
    Conv2D(3×3, ReLU) × 2 → Dropout
    Channels: 1024

Expansive path (decoder):
    4 × DecoderBlock → [Conv2DTranspose(2×2) → Concat(skip) → Conv2D(3×3, ReLU) × 2]
    Channels: 512 → 256 → 128 → 64

Output head:
    Conv2D(1×1, softmax, filters=3)

Total parameters (default): ~31 M — suitable for a single 8 GB GPU
with batch_size=8 and patch_size=256.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model


# ─── building blocks ──────────────────────────────────────────────────────────

def _conv_block(
    x: tf.Tensor,
    filters: int,
    dropout_rate: float = 0.0,
    name: str = "conv_block",
) -> tf.Tensor:
    """Two stacked Conv2D(3×3, 'same', ReLU) layers followed by optional Dropout.

    BatchNormalisation is intentionally omitted here: BN interacts poorly
    with small batch sizes on large-patch inputs because the batch statistics
    become noisy.  Dropout provides sufficient regularisation.

    Args:
        x:            Input feature map tensor.
        filters:      Number of convolutional filters.
        dropout_rate: Spatial dropout rate (0.0 = no dropout).
        name:         Block name prefix for graph readability.

    Returns:
        Output feature map tensor.
    """
    x = layers.Conv2D(
        filters, kernel_size=3, padding="same", activation="relu",
        kernel_initializer="he_normal", name=f"{name}_conv1",
    )(x)
    x = layers.Conv2D(
        filters, kernel_size=3, padding="same", activation="relu",
        kernel_initializer="he_normal", name=f"{name}_conv2",
    )(x)
    if dropout_rate > 0.0:
        # SpatialDropout2D drops entire feature maps — better than element-wise
        # dropout for convolutional features (Tompson et al., 2015).
        x = layers.SpatialDropout2D(dropout_rate, name=f"{name}_drop")(x)
    return x


def _encoder_block(
    x: tf.Tensor,
    filters: int,
    dropout_rate: float = 0.1,
    name: str = "encoder",
) -> tuple[tf.Tensor, tf.Tensor]:
    """One contracting step: conv_block → MaxPool2D.

    Returns both the skip-connection tensor (pre-pool) and the downsampled
    tensor that propagates further down the encoder.

    Args:
        x:            Input tensor.
        filters:      Number of conv filters.
        dropout_rate: Dropout rate inside the conv block.
        name:         Block name prefix.

    Returns:
        (skip, pooled) tuple of tensors.
    """
    skip = _conv_block(x, filters, dropout_rate=dropout_rate, name=name)
    pooled = layers.MaxPooling2D(pool_size=2, strides=2, name=f"{name}_pool")(skip)
    return skip, pooled


def _decoder_block(
    x: tf.Tensor,
    skip: tf.Tensor,
    filters: int,
    dropout_rate: float = 0.1,
    name: str = "decoder",
) -> tf.Tensor:
    """One expansive step: Conv2DTranspose → Concatenate(skip) → conv_block.

    Using Conv2DTranspose (learnable upsampling) rather than bilinear upsampling
    + Conv2D gives the model freedom to learn its own upsampling kernel, which
    matters for irregular cloud shapes.

    Args:
        x:            Input tensor from the previous decoder stage.
        skip:         Corresponding encoder skip-connection tensor.
        filters:      Number of conv filters in this stage.
        dropout_rate: Dropout rate inside the conv block.
        name:         Block name prefix.

    Returns:
        Output tensor after upsampling, concatenation, and convolution.
    """
    x = layers.Conv2DTranspose(
        filters, kernel_size=2, strides=2, padding="same",
        kernel_initializer="he_normal", name=f"{name}_upsample",
    )(x)
    # Guard against spatial dimension mismatch caused by odd input sizes.
    if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
        x = layers.Resizing(
            skip.shape[1], skip.shape[2], name=f"{name}_resize"
        )(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = _conv_block(x, filters, dropout_rate=dropout_rate, name=name)
    return x


# ─── full U-Net model ─────────────────────────────────────────────────────────

def build_unet(
    input_shape: tuple[int, int, int] = (256, 256, 4),
    num_classes: int = 3,
    base_filters: int = 64,
    depth: int = 4,
    dropout_rate: float = 0.1,
    bottleneck_dropout: float = 0.3,
) -> Model:
    """Build and return a multi-class U-Net Keras model.

    The architecture depth and filter count are parameterised so you can
    trade model capacity for GPU memory.  Default values target a single
    8 GB GPU with 256×256 patches and batch_size=8.

    Args:
        input_shape:         (H, W, channels) — default (256, 256, 4).
        num_classes:         Number of output segmentation classes (default 3).
        base_filters:        Filters in the first encoder block; each subsequent
                             block doubles this (default 64).
        depth:               Number of encoder/decoder stages (default 4).
        dropout_rate:        Dropout in encoder/decoder conv blocks (default 0.1).
        bottleneck_dropout:  Higher dropout in the bottleneck (default 0.3).

    Returns:
        Compiled-ready Keras Model (not yet compiled — compilation is done
        in the training module to keep architecture and training concerns
        separate).
    """
    inputs = layers.Input(shape=input_shape, name="rgbnir_input")

    # ── Encoder ──────────────────────────────────────────────────────────────
    skips: list[tf.Tensor] = []
    x = inputs
    for level in range(depth):
        filters = base_filters * (2 ** level)
        skip, x = _encoder_block(
            x, filters,
            dropout_rate=dropout_rate,
            name=f"enc{level + 1}",
        )
        skips.append(skip)

    # ── Bottleneck ───────────────────────────────────────────────────────────
    bottleneck_filters = base_filters * (2 ** depth)
    x = _conv_block(
        x, bottleneck_filters,
        dropout_rate=bottleneck_dropout,
        name="bottleneck",
    )

    # ── Decoder ──────────────────────────────────────────────────────────────
    for level in reversed(range(depth)):
        filters = base_filters * (2 ** level)
        x = _decoder_block(
            x, skips[level],
            filters=filters,
            dropout_rate=dropout_rate,
            name=f"dec{level + 1}",
        )

    # ── Output head ──────────────────────────────────────────────────────────
    # 1×1 convolution collapses the feature depth to num_classes probability
    # maps; softmax ensures per-pixel probabilities sum to 1.
    outputs = layers.Conv2D(
        num_classes,
        kernel_size=1,
        padding="same",
        activation="softmax",
        dtype="float32",          # Keep output in float32 even with mixed precision
        name="segmentation_output",
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="CloudShadow_UNet")
    return model


# ─── model summary utility ────────────────────────────────────────────────────

def model_summary(input_shape: tuple[int, int, int] = (256, 256, 4)) -> None:
    """Print a parameter summary for the default U-Net configuration."""
    model = build_unet(input_shape=input_shape)
    model.summary(line_length=100)


if __name__ == "__main__":
    model_summary()
