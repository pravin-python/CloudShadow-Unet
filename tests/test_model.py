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

def test_dice_coefficient_edge_cases():
    metric = DiceCoefficient()

    # Edge case 1: All zeros (e.g., testing smoothing/division by zero avoidance)
    y_true_zeros = np.zeros((1, 2, 2, 3), dtype=np.float32)
    y_pred_zeros = np.zeros((1, 2, 2, 3), dtype=np.float32)

    metric.update_state(y_true_zeros, y_pred_zeros)
    assert np.isclose(metric.result().numpy(), 1.0, atol=1e-5)

    metric.reset_state()

    # Edge case 2: Disjoint/completely incorrect predictions across all classes
    # y_true has 2 pixels class 0, 1 pixel class 1, 1 pixel class 2
    y_true_disjoint = np.array([
        [[[1, 0, 0], [0, 1, 0]],
         [[0, 0, 1], [1, 0, 0]]]
    ], dtype=np.float32)

    # y_pred is entirely disjoint from y_true, with all classes present
    y_pred_disjoint = np.array([
        [[[0, 1, 0], [0, 0, 1]],
         [[1, 0, 0], [0, 1, 0]]]
    ], dtype=np.float32)

    metric.update_state(y_true_disjoint, y_pred_disjoint)
    assert np.isclose(metric.result().numpy(), 0.0, atol=1e-5)
