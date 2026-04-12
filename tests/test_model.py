import pytest
import numpy as np
import tensorflow as tf
from model import MultiClassDiceLoss, DiceCoefficient, MeanIoU, CombinedDiceCELoss, _conv_relu_block

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

def test_conv_relu_block_no_dropout():
    """Test _conv_relu_block output shape and layers without dropout."""
    inputs = tf.keras.Input(shape=(256, 256, 4))
    outputs = _conv_relu_block(inputs, filters=64, dropout=0.0, name="test_block")
    model = tf.keras.Model(inputs, outputs)

    # 2 conv layers + 1 input layer
    assert len(model.layers) == 3

    # Assert layer types
    layer_types = [type(layer) for layer in model.layers]
    assert tf.keras.layers.Conv2D in layer_types
    assert tf.keras.layers.SpatialDropout2D not in layer_types

    # Verify shape
    assert model.output_shape == (None, 256, 256, 64)

def test_conv_relu_block_with_dropout():
    """Test _conv_relu_block output shape and layers with dropout."""
    inputs = tf.keras.Input(shape=(128, 128, 64))
    outputs = _conv_relu_block(inputs, filters=128, dropout=0.5, name="test_block_drop")
    model = tf.keras.Model(inputs, outputs)

    # 2 conv layers + 1 spatial dropout + 1 input layer
    assert len(model.layers) == 4

    # Assert layer types
    layer_types = [type(layer) for layer in model.layers]
    assert tf.keras.layers.Conv2D in layer_types
    assert tf.keras.layers.SpatialDropout2D in layer_types

    # Verify shape
    assert model.output_shape == (None, 128, 128, 128)

def test_mean_iou_metric():
    metric = MeanIoU()

    y_true = np.array([
        [[[1, 0, 0], [0, 1, 0]]]
    ], dtype=np.float32)
    y_pred = y_true.copy()

    metric.update_state(y_true, y_pred)
    assert np.isclose(metric.result().numpy(), 1.0)
