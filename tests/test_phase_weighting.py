"""
Tests for the phase weighting module.
"""

import numpy as np
import pytest
from src.phase_weighting import (
    PHI,
    compute_phase_weights,
    apply_phase_weights,
    sweep_k_values
)


def test_golden_ratio_constant():
    """Test that PHI is approximately the golden ratio."""
    assert np.isclose(PHI, 1.618033988749895, rtol=1e-10)


def test_compute_phase_weights_shape():
    """Test that phase weights have the correct shape."""
    n = 20
    weights = compute_phase_weights(n, k=0.3)
    
    assert len(weights) == n
    assert weights.dtype == np.float64


def test_compute_phase_weights_positive():
    """Test that all phase weights are non-negative."""
    weights = compute_phase_weights(20, k=0.3)
    
    assert np.all(weights >= 0)  # Allow zero for when n mod φ == 0


def test_compute_phase_weights_different_k():
    """Test that different k values produce different weights."""
    weights_k03 = compute_phase_weights(20, k=0.3)
    weights_k05 = compute_phase_weights(20, k=0.5)
    
    # Should be different
    assert not np.allclose(weights_k03, weights_k05)


def test_apply_phase_weights_shape():
    """Test that applying weights preserves spectrum shape."""
    spectrum = np.array([1+1j, 2+2j, 3+3j, 4+4j])
    weighted = apply_phase_weights(spectrum, k=0.3)
    
    assert len(weighted) == len(spectrum)
    assert weighted.dtype == np.complex128


def test_apply_phase_weights_modifies():
    """Test that phase weighting actually modifies the spectrum."""
    spectrum = np.array([1+1j, 2+2j, 3+3j, 4+4j])
    weighted = apply_phase_weights(spectrum, k=0.3)
    
    # Should be different from original
    assert not np.allclose(weighted, spectrum)


def test_sweep_k_values():
    """Test k-value sweep."""
    spectrum = np.random.randn(20) + 1j * np.random.randn(20)
    results = sweep_k_values(spectrum, k_min=0.2, k_max=0.4, k_step=0.1)
    
    # Should have 3 k values: 0.2, 0.3, 0.4 (with floating point tolerance)
    assert len(results) == 3
    
    # Check that keys are approximately correct
    keys = sorted(results.keys())
    assert np.isclose(keys[0], 0.2, atol=1e-10)
    assert np.isclose(keys[1], 0.3, atol=1e-10)
    assert np.isclose(keys[2], 0.4, atol=1e-10)
    
    # Each should be an array of the same length
    for k, weighted_spectrum in results.items():
        assert len(weighted_spectrum) == len(spectrum)
