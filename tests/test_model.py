import pytest
import numpy as np
import tensorflow as tf
from model import MultiClassDiceLoss, DiceCoefficient, MeanIoU, CombinedDiceCELoss, configure_precision

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


def test_configure_precision_invalid():
    with pytest.raises(ValueError, match="policy must be one of"):
        configure_precision("invalid_policy")

def test_configure_precision_explicit(mocker):
    # Mock set_global_policy so we don't actually change the state in tests
    mock_set = mocker.patch("tensorflow.keras.mixed_precision.set_global_policy")

    assert configure_precision("bfloat16") == "bfloat16"
    mock_set.assert_called_with("bfloat16")

    assert configure_precision("float16") == "float16"
    mock_set.assert_called_with("float16")

    assert configure_precision("float32") == "float32"
    mock_set.assert_called_with("float32")

def test_configure_precision_auto_no_gpu(mocker):
    mock_set = mocker.patch("tensorflow.keras.mixed_precision.set_global_policy")
    mocker.patch("tensorflow.config.list_physical_devices", return_value=[])

    assert configure_precision("auto") == "float32"
    mock_set.assert_called_with("float32")

def test_configure_precision_auto_gpu_ampere(mocker):
    mock_set = mocker.patch("tensorflow.keras.mixed_precision.set_global_policy")
    mocker.patch("tensorflow.config.list_physical_devices", return_value=["GPU:0"])
    mocker.patch("tensorflow.config.experimental.get_device_details", return_value={"compute_capability": (8, 0)})

    assert configure_precision("auto") == "bfloat16"
    mock_set.assert_called_with("bfloat16")

def test_configure_precision_auto_gpu_turing(mocker):
    mock_set = mocker.patch("tensorflow.keras.mixed_precision.set_global_policy")
    mocker.patch("tensorflow.config.list_physical_devices", return_value=["GPU:0"])
    mocker.patch("tensorflow.config.experimental.get_device_details", return_value={"compute_capability": (7, 5)})

    assert configure_precision("auto") == "float16"
    mock_set.assert_called_with("float16")

def test_configure_precision_auto_gpu_older(mocker):
    mock_set = mocker.patch("tensorflow.keras.mixed_precision.set_global_policy")
    mocker.patch("tensorflow.config.list_physical_devices", return_value=["GPU:0"])
    mocker.patch("tensorflow.config.experimental.get_device_details", return_value={"compute_capability": (6, 0)})

    assert configure_precision("auto") == "float32"
    mock_set.assert_called_with("float32")

def test_configure_precision_auto_exception(mocker):
    mock_set = mocker.patch("tensorflow.keras.mixed_precision.set_global_policy")
    mocker.patch("tensorflow.config.list_physical_devices", return_value=["GPU:0"])
    mocker.patch("tensorflow.config.experimental.get_device_details", side_effect=Exception("Failed to get details"))

    assert configure_precision("auto") == "float16"
    mock_set.assert_called_with("float16")
