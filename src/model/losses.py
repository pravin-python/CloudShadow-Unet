"""
Module 4 — Specialized Loss Functions & Metrics
================================================
Implements the Multi-Class Dice Loss and the Dice Coefficient / IoU
(Jaccard Index) metrics used to train and evaluate the U-Net.

Why not standard Categorical Cross-Entropy?
-------------------------------------------
The 38-Cloud dataset (and satellite cloud datasets in general) exhibit
extreme class imbalance:
    Background ~80–90% of pixels
    Cloud      ~5–15%
    Shadow     ~1–5%

CCE gradient is dominated by the background class, causing the model to
predict "all background" and achieve ~85% pixel accuracy while completely
failing on clouds/shadows.

Dice Loss optimises region *overlap* rather than pixel counts, making it
invariant to class frequency.  The loss for each class is:

    DiceLoss_c = 1 - (2 * Σ(y_true * y_pred) + ε) / (Σ y_true + Σ y_pred + ε)

The Multi-Class Dice Loss is the macro-averaged Dice Loss across all classes:

    MultiDiceLoss = mean(DiceLoss_c  for c in classes)

In practice we combine Dice Loss with a small weight of CCE to preserve
pixel-level gradient signal, especially during early training epochs when
predictions are near-uniform.

Combined Loss = α × DiceLoss + (1-α) × CategoricalCrossEntropy
    default α = 0.7
"""

from __future__ import annotations

import tensorflow as tf


# ─── smoothing constant ───────────────────────────────────────────────────────
# Prevents division-by-zero when a class is absent from a batch.
_SMOOTH: float = 1e-6


# ─── per-class Dice coefficient ───────────────────────────────────────────────

def dice_coefficient_per_class(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:
    """Compute per-class Dice coefficient.

    Both inputs are expected in one-hot / softmax format:
        shape = (batch, H, W, num_classes).

    The spatial dimensions (H, W) are reduced; the batch dimension is
    averaged.  Returns a (num_classes,) tensor of Dice scores.

    Args:
        y_true: Ground-truth one-hot tensor.
        y_pred: Predicted softmax probability tensor.

    Returns:
        (num_classes,) float32 tensor of per-class Dice scores ∈ [0, 1].
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Sum over spatial dims (H, W), keep batch and class axes
    axes = [1, 2]
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)   # (B, C)
    sum_true = tf.reduce_sum(y_true, axis=axes)                 # (B, C)
    sum_pred = tf.reduce_sum(y_pred, axis=axes)                 # (B, C)

    dice_per_sample = (2.0 * intersection + _SMOOTH) / (sum_true + sum_pred + _SMOOTH)
    # Average over batch dimension → (C,)
    return tf.reduce_mean(dice_per_sample, axis=0)


# ─── loss functions ───────────────────────────────────────────────────────────

def multiclass_dice_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:
    """Macro-averaged Multi-Class Dice Loss.

    Loss ∈ [0, 1] where 0 = perfect overlap for all classes.

    Args:
        y_true: One-hot ground truth (batch, H, W, num_classes).
        y_pred: Softmax predictions  (batch, H, W, num_classes).

    Returns:
        Scalar loss tensor.
    """
    dice_per_class = dice_coefficient_per_class(y_true, y_pred)
    # Macro average: treat each class equally regardless of frequency
    mean_dice = tf.reduce_mean(dice_per_class)
    return 1.0 - mean_dice


def combined_dice_ce_loss(
    alpha: float = 0.7,
) -> "CombinedLoss":
    """Factory that returns a Combined Dice + Cross-Entropy loss callable.

    Args:
        alpha: Weight on Dice Loss component. (1-alpha) is applied to CCE.
               Default 0.7 (70% Dice, 30% CCE).

    Returns:
        A callable loss function compatible with model.compile(loss=...).
    """
    return CombinedLoss(alpha=alpha)


class CombinedLoss(tf.keras.losses.Loss):
    """Weighted sum of Multi-Class Dice Loss and Categorical Cross-Entropy.

    Attributes:
        alpha: Dice Loss weight (CCE weight = 1 - alpha).
    """

    def __init__(self, alpha: float = 0.7, name: str = "combined_dice_ce") -> None:
        super().__init__(name=name)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
        self.alpha = alpha
        self._cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute combined loss.

        Args:
            y_true: One-hot ground truth.
            y_pred: Softmax predictions.

        Returns:
            Scalar loss tensor.
        """
        dice = multiclass_dice_loss(y_true, y_pred)
        cce = self._cce(y_true, y_pred)
        return self.alpha * dice + (1.0 - self.alpha) * cce

    def get_config(self) -> dict:
        base = super().get_config()
        base["alpha"] = self.alpha
        return base


# ─── metrics ──────────────────────────────────────────────────────────────────

class DiceCoefficient(tf.keras.metrics.Metric):
    """Macro-averaged Dice Coefficient metric (higher is better).

    Accumulates per-batch intersections and sums to compute an
    epoch-level running average instead of averaging per-batch scores,
    which would be biased for small or class-absent batches.
    """

    def __init__(self, num_classes: int = 3, name: str = "dice_coeff") -> None:
        super().__init__(name=name)
        self.num_classes = num_classes
        # Accumulators for numerator and denominator per class
        self._intersection = self.add_weight(
            "intersection", shape=(num_classes,), initializer="zeros"
        )
        self._sum_true = self.add_weight(
            "sum_true", shape=(num_classes,), initializer="zeros"
        )
        self._sum_pred = self.add_weight(
            "sum_pred", shape=(num_classes,), initializer="zeros"
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight=None,
    ) -> None:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        axes = [0, 1, 2]  # reduce batch + spatial dims
        self._intersection.assign_add(tf.reduce_sum(y_true * y_pred, axis=axes))
        self._sum_true.assign_add(tf.reduce_sum(y_true, axis=axes))
        self._sum_pred.assign_add(tf.reduce_sum(y_pred, axis=axes))

    def result(self) -> tf.Tensor:
        dice_per_class = (2.0 * self._intersection + _SMOOTH) / (
            self._sum_true + self._sum_pred + _SMOOTH
        )
        return tf.reduce_mean(dice_per_class)

    def reset_state(self) -> None:
        self._intersection.assign(tf.zeros(self.num_classes))
        self._sum_true.assign(tf.zeros(self.num_classes))
        self._sum_pred.assign(tf.zeros(self.num_classes))

    def get_config(self) -> dict:
        base = super().get_config()
        base["num_classes"] = self.num_classes
        return base


class MeanIoU(tf.keras.metrics.Metric):
    """Mean Intersection-over-Union (Jaccard Index) for one-hot predictions.

    Unlike the built-in tf.keras.metrics.MeanIoU, this implementation
    works directly with softmax probability outputs (no argmax required in
    the metric itself) by accepting one-hot targets and threshold-argmaxed
    predictions, giving identical results with cleaner code.
    """

    def __init__(self, num_classes: int = 3, name: str = "mean_iou") -> None:
        super().__init__(name=name)
        self.num_classes = num_classes
        self._intersection = self.add_weight(
            "intersection", shape=(num_classes,), initializer="zeros"
        )
        self._union = self.add_weight(
            "union", shape=(num_classes,), initializer="zeros"
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight=None,
    ) -> None:
        y_true = tf.cast(y_true, tf.float32)

        # Convert softmax output to hard one-hot via argmax
        pred_labels = tf.argmax(y_pred, axis=-1)
        y_pred_onehot = tf.one_hot(pred_labels, depth=self.num_classes)
        y_pred_onehot = tf.cast(y_pred_onehot, tf.float32)

        axes = [0, 1, 2]
        intersection = tf.reduce_sum(y_true * y_pred_onehot, axis=axes)
        union = tf.reduce_sum(y_true + y_pred_onehot, axis=axes) - intersection

        self._intersection.assign_add(intersection)
        self._union.assign_add(union)

    def result(self) -> tf.Tensor:
        iou_per_class = (self._intersection + _SMOOTH) / (self._union + _SMOOTH)
        return tf.reduce_mean(iou_per_class)

    def reset_state(self) -> None:
        self._intersection.assign(tf.zeros(self.num_classes))
        self._union.assign(tf.zeros(self.num_classes))

    def get_config(self) -> dict:
        base = super().get_config()
        base["num_classes"] = self.num_classes
        return base
