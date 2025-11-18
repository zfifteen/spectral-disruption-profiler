"""
Phase weighting module implementing θ′(n,k) phase weighting with golden ratio geometry.

Phase weighting formula: θ′(n,k) = φ · ((n mod φ)/φ)^k
where φ ≈ 1.618033988749895 (golden ratio)
"""

import numpy as np
from typing import Union


# Golden ratio constant
PHI = (1 + np.sqrt(5)) / 2


def compute_phase_weights(n: int, k: float = 0.300) -> np.ndarray:
    """
    Compute θ′(n,k) phase weights for spectral analysis.
    
    Args:
        n: Sequence length or number of frequency bins
        k: Phase weighting exponent (default: 0.300, the hypothesized optimal value)
        
    Returns:
        Array of phase weights
    """
    indices = np.arange(n)
    
    # θ′(n,k) = φ · ((n mod φ)/φ)^k
    phase_weights = PHI * np.power((indices % PHI) / PHI, k)
    
    return phase_weights


def apply_phase_weights(spectrum: np.ndarray, k: float = 0.300) -> np.ndarray:
    """
    Apply θ′(n,k) phase weighting to a frequency spectrum.
    
    Args:
        spectrum: Complex frequency spectrum from FFT
        k: Phase weighting exponent
        
    Returns:
        Phase-weighted spectrum
    """
    n = len(spectrum)
    weights = compute_phase_weights(n, k)
    
    # Apply weights to spectrum magnitudes while preserving phase
    magnitudes = np.abs(spectrum)
    phases = np.angle(spectrum)
    
    weighted_magnitudes = magnitudes * weights
    
    return weighted_magnitudes * np.exp(1j * phases)


def sweep_k_values(spectrum: np.ndarray, 
                   k_min: float = 0.20, 
                   k_max: float = 0.40, 
                   k_step: float = 0.01) -> dict:
    """
    Sweep through k values and return weighted spectra for each.
    
    Args:
        spectrum: Complex frequency spectrum
        k_min: Minimum k value
        k_max: Maximum k value
        k_step: Step size for k sweep
        
    Returns:
        Dictionary mapping k values to weighted spectra
    """
    k_values = np.arange(k_min, k_max + k_step, k_step)
    results = {}
    
    for k in k_values:
        results[float(k)] = apply_phase_weights(spectrum, k)
    
    return results
