import pytest
import numpy as np
import tensorflow as tf
from src.model.losses import dice_coefficient_per_class

def test_dice_coefficient_per_class_perfect_prediction():
    # Batch size 1, 2x2 spatial, 3 classes
    y_true = np.array([
        [[[1, 0, 0], [0, 1, 0]],
         [[0, 0, 1], [1, 0, 0]]]
    ], dtype=np.float32)
    y_pred = y_true.copy()

    result = dice_coefficient_per_class(tf.constant(y_true), tf.constant(y_pred))
    assert result.shape == (3,)
    assert np.allclose(result.numpy(), [1.0, 1.0, 1.0], atol=1e-5)

def test_dice_coefficient_per_class_disjoint_prediction():
    y_true = np.array([
        [[[1, 0, 0], [1, 0, 0]],
         [[1, 0, 0], [1, 0, 0]]]
    ], dtype=np.float32)
    y_pred = np.array([
        [[[0, 1, 0], [0, 1, 0]],
         [[0, 1, 0], [0, 1, 0]]]
    ], dtype=np.float32)

    result = dice_coefficient_per_class(tf.constant(y_true), tf.constant(y_pred))
    # y_true has class 0, y_pred has class 1. Class 2 is 0 for both.
    # Class 0: y_true sum=4, y_pred sum=0, inter=0 -> dice=0
    # Class 1: y_true sum=0, y_pred sum=4, inter=0 -> dice=0
    # Class 2: y_true sum=0, y_pred sum=0, inter=0 -> dice=1.0 (because of smooth)
    res_np = result.numpy()
    assert np.isclose(res_np[0], 0.0, atol=1e-5)
    assert np.isclose(res_np[1], 0.0, atol=1e-5)
    assert np.isclose(res_np[2], 1.0, atol=1e-5)

def test_dice_coefficient_per_class_partial_overlap():
    # Batch size 2 to test batch averaging
    y_true = np.array([
        # Sample 1
        [[[1, 0, 0], [1, 0, 0]],
         [[0, 1, 0], [0, 1, 0]]],
        # Sample 2
        [[[0, 0, 1], [0, 0, 1]],
         [[1, 0, 0], [1, 0, 0]]]
    ], dtype=np.float32)
    y_pred = np.array([
        # Sample 1: Class 0 is 50% overlap, Class 1 is 50% overlap
        [[[1, 0, 0], [0, 1, 0]],
         [[1, 0, 0], [0, 1, 0]]],
        # Sample 2: Class 2 is perfect, Class 0 is wrong
        [[[0, 0, 1], [0, 0, 1]],
         [[0, 1, 0], [0, 1, 0]]]
    ], dtype=np.float32)

    result = dice_coefficient_per_class(tf.constant(y_true), tf.constant(y_pred))

    res_np = result.numpy()
    assert np.isclose(res_np[0], 0.25, atol=1e-5)
    assert np.isclose(res_np[1], 0.25, atol=1e-5)
    assert np.isclose(res_np[2], 1.0, atol=1e-5)

def test_dice_coefficient_per_class_different_types():
    # Test handling of different tensor types (should be cast to float32)
    y_true = np.array([
        [[[1, 0], [0, 1]]]
    ], dtype=np.int32)
    y_pred = np.array([
        [[[1, 0], [0, 1]]]
    ], dtype=np.float64)

    result = dice_coefficient_per_class(tf.constant(y_true), tf.constant(y_pred))
    assert result.dtype == tf.float32
    assert np.allclose(result.numpy(), [1.0, 1.0], atol=1e-5)
