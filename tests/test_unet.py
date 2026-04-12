import pytest
import tensorflow as tf
import numpy as np

from src.model.unet import build_unet, _conv_block, _encoder_block, _decoder_block

def test_conv_block():
    """Verify that _conv_block returns a tensor of correct shape."""
    inputs = tf.keras.Input(shape=(128, 128, 64))
    outputs = _conv_block(inputs, filters=128)

    # Should maintain spatial dims, update channels
    assert outputs.shape[1:] == (128, 128, 128)

def test_encoder_block():
    """Verify that _encoder_block returns correctly sized skip and pooled tensors."""
    inputs = tf.keras.Input(shape=(128, 128, 64))
    skip, pooled = _encoder_block(inputs, filters=128)

    # Skip maintains spatial dims, pooled halves them
    assert skip.shape[1:] == (128, 128, 128)
    assert pooled.shape[1:] == (64, 64, 128)

def test_decoder_block():
    """Verify that _decoder_block correctly upsamples and concats."""
    inputs = tf.keras.Input(shape=(64, 64, 256))
    skip = tf.keras.Input(shape=(128, 128, 128))

    outputs = _decoder_block(inputs, skip, filters=128)

    # Decoded output should match the skip connection's spatial dims and the block's filter count
    assert outputs.shape[1:] == (128, 128, 128)

def test_build_unet_default_shapes():
    """Verify the default U-Net architecture input/output shapes."""
    model = build_unet()

    assert model.input_shape == (None, 256, 256, 4)
    assert model.output_shape == (None, 256, 256, 3)

    # Total params should be roughly 31M for the default configuration
    # We check that it's in the expected range.
    assert 31_000_000 < model.count_params() < 32_000_000

def test_build_unet_custom_shapes():
    """Verify the U-Net architecture builds correctly with custom params."""
    model = build_unet(input_shape=(128, 128, 8), num_classes=5, base_filters=32)

    assert model.input_shape == (None, 128, 128, 8)
    assert model.output_shape == (None, 128, 128, 5)

def test_unet_output_softmax():
    """Verify the model outputs a valid probability distribution (sums to 1)."""
    model = build_unet(input_shape=(64, 64, 4))

    # Generate dummy input data
    dummy_input = np.random.rand(1, 64, 64, 4).astype(np.float32)

    # Get predictions
    preds = model.predict(dummy_input)

    # Check that probabilities sum to 1 over the last axis
    sums = np.sum(preds, axis=-1)

    # Allow a small tolerance for floating point arithmetic
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5, atol=1e-5)
