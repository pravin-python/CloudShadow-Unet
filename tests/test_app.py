import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app import _apply_confidence_threshold

def test_apply_confidence_threshold_no_threshold():
    class_map = np.array([[1, 2], [0, 1]])
    result = _apply_confidence_threshold(class_map, 0.0, b'data', 'model.h5', 256, 0.25)
    np.testing.assert_array_equal(result, class_map)

@patch('app._read_geotiff_bytes')
def test_apply_confidence_threshold_no_image(mock_read_geotiff_bytes):
    mock_read_geotiff_bytes.return_value = None
    class_map = np.array([[1, 2], [0, 1]])
    result = _apply_confidence_threshold(class_map, 0.5, b'data', 'model.h5', 256, 0.25)
    np.testing.assert_array_equal(result, class_map)

@patch('app._read_geotiff_bytes')
@patch('app._load_model')
def test_apply_confidence_threshold_no_model(mock_load_model, mock_read_geotiff_bytes):
    mock_read_geotiff_bytes.return_value = (np.zeros((10, 10, 4)), {})
    mock_load_model.return_value = None
    class_map = np.array([[1, 2], [0, 1]])
    result = _apply_confidence_threshold(class_map, 0.5, b'data', 'model.h5', 256, 0.25)
    np.testing.assert_array_equal(result, class_map)

@patch('app._read_geotiff_bytes')
@patch('app._load_model')
@patch('geospatial_utils.generate_tile_coords')
@patch('geospatial_utils.cosine_bell_mask')
def test_apply_confidence_threshold_with_threshold(mock_bell, mock_coords, mock_load_model, mock_read_geotiff_bytes):
    # Setup mock image
    mock_read_geotiff_bytes.return_value = (np.zeros((2, 2, 4)), {})

    # Setup mock model
    mock_model = MagicMock()
    # Predictions matching tile coords (we'll return a batch of 1 patch)
    # Probabilities:
    # (0, 0): [0.1, 0.8, 0.1] -> Max prob 0.8, label 1
    # (0, 1): [0.1, 0.1, 0.8] -> Max prob 0.8, label 2
    # (1, 0): [0.4, 0.4, 0.2] -> Max prob 0.4, label 0
    # (1, 1): [0.3, 0.4, 0.3] -> Max prob 0.4, label 1
    mock_preds = np.array([[[[0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
                            [[0.4, 0.4, 0.2], [0.3, 0.4, 0.3]]]])
    mock_model.predict.return_value = mock_preds
    mock_load_model.return_value = mock_model

    # Setup mock coords and bell
    mock_coords.return_value = [(0, 2, 0, 2)]
    mock_bell.return_value = np.ones((2, 2))

    class_map = np.array([[1, 2], [0, 1]])

    # Test with threshold 0.5
    # (1, 0) and (1, 1) have max probs < 0.5 (0.4), so they should become 0
    result = _apply_confidence_threshold(class_map, 0.5, b'data', 'model.h5', 2, 0.0)

    expected = np.array([[1, 2], [0, 0]])
    np.testing.assert_array_equal(result, expected)
