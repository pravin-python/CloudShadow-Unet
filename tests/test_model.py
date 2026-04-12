import pytest
import numpy as np
import tensorflow as tf
from model import MultiClassDiceLoss, DiceCoefficient, MeanIoU, CombinedDiceCELoss

def test_multi_class_dice_loss():
    loss_fn = MultiClassDiceLoss()

    # Perfect prediction
    y_true = np.array([
        [[[1, 0, 0], [0, 1, 0]],
         [[0, 0, 1], [1, 0, 0]]]
    ], dtype=np.float32)
    y_pred = y_true.copy()

    loss = loss_fn(y_true, y_pred)
    assert np.isclose(loss.numpy(), 0.0, atol=1e-5)

    # Completely wrong prediction
    y_pred_wrong = np.array([
        [[[0, 1, 0], [1, 0, 0]],
         [[1, 0, 0], [0, 1, 0]]]
    ], dtype=np.float32)

    loss_wrong = loss_fn(y_true, y_pred_wrong)
    assert loss_wrong.numpy() > 0.9

def test_combined_loss():
    loss_fn = CombinedDiceCELoss(alpha=0.5)

    y_true = np.array([
        [[[1, 0, 0], [0, 1, 0]],
         [[0, 0, 1], [1, 0, 0]]]
    ], dtype=np.float32)
    y_pred = y_true.copy()

    loss = loss_fn(y_true, y_pred)
    assert np.isclose(loss.numpy(), 0.0, atol=1e-5)

def test_dice_coefficient_metric():
    metric = DiceCoefficient()

    y_true = np.array([
        [[[1, 0, 0], [0, 1, 0]]]
    ], dtype=np.float32)
    y_pred = y_true.copy()

    metric.update_state(y_true, y_pred)
    assert np.isclose(metric.result().numpy(), 1.0)

def test_mean_iou_metric():
    metric = MeanIoU()

    y_true = np.array([
        [[[1, 0, 0], [0, 1, 0]]]
    ], dtype=np.float32)
    y_pred = y_true.copy()

    metric.update_state(y_true, y_pred)
    assert np.isclose(metric.result().numpy(), 1.0)
