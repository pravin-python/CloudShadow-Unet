import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.inference.predict import _cosine_bell_mask

def test_cosine_bell_mask_shape():
    """Test that the generated mask has the correct 2D shape."""
    mask = _cosine_bell_mask(256)
    assert mask.shape == (256, 256)

def test_cosine_bell_mask_values():
    """Test that the generated mask contains values in the expected range."""
    mask = _cosine_bell_mask(100)
    assert np.all(mask > 0.0), "Mask should not contain zeros"
    assert np.all(mask <= 1.0 + 1e-6), "Mask should not exceed 1.0 (plus epsilon)"
    assert np.isclose(mask.max(), 1.0 + 1e-6), "Peak value should be 1.0 (plus epsilon)"

def test_cosine_bell_mask_symmetry():
    """Test that the generated mask is radially symmetric."""
    mask = _cosine_bell_mask(128)
    assert np.allclose(mask, mask.T), "Mask should be symmetric (transpose equals itself)"
    assert np.allclose(mask, np.flipud(mask)), "Mask should be symmetric up/down"
    assert np.allclose(mask, np.fliplr(mask)), "Mask should be symmetric left/right"
