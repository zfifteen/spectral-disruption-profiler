"""
Spectral features extraction module using FFT-based analysis.

Computes spectral disruption scores and related features for gRNA sequences.
"""

import numpy as np
from scipy import fft
from typing import Dict, List, Tuple, Optional
from .encoding import encode_sequence, pad_sequence
from .phase_weighting import apply_phase_weights


def compute_fft_spectrum(encoded_sequence: np.ndarray) -> np.ndarray:
    """
    Compute FFT spectrum of an encoded DNA sequence.
    
    Args:
        encoded_sequence: Complex encoded DNA sequence
        
    Returns:
        Complex FFT spectrum
    """
    return fft.fft(encoded_sequence)


def compute_spectral_disruption_score(sequence: str, 
                                      k: float = 0.300,
                                      use_phase_weighting: bool = True) -> float:
    """
    Compute spectral disruption score for a gRNA sequence.
    
    This is the core scoring function that applies FFT encoding,
    optional phase weighting, and returns a scalar score.
    
    Args:
        sequence: DNA sequence string
        k: Phase weighting exponent (if use_phase_weighting=True)
        use_phase_weighting: Whether to apply θ′(n,k) phase weighting
        
    Returns:
        Spectral disruption score (higher = more disruptive/effective)
    """
    # Encode sequence
    encoded = encode_sequence(sequence)
    
    # Compute FFT
    spectrum = compute_fft_spectrum(encoded)
    
    # Apply phase weighting if requested
    if use_phase_weighting:
        spectrum = apply_phase_weights(spectrum, k)
    
    # Compute score as the sum of magnitude-weighted frequency components
    # Focus on lower frequency components (first half of spectrum)
    half_len = len(spectrum) // 2
    magnitudes = np.abs(spectrum[:half_len])
    
    # Weight by inverse frequency (DC and low frequencies matter more)
    freq_weights = 1.0 / (np.arange(half_len) + 1)
    
    score = np.sum(magnitudes * freq_weights)
    
    # Normalize by sequence length
    score = score / len(sequence)
    
    return float(score)


def compute_spectral_features(sequence: str, k: float = 0.300) -> Dict[str, float]:
    """
    Compute a comprehensive set of spectral features for a sequence.
    
    Args:
        sequence: DNA sequence string
        k: Phase weighting exponent
        
    Returns:
        Dictionary of feature names to values
    """
    encoded = encode_sequence(sequence)
    spectrum = compute_fft_spectrum(encoded)
    weighted_spectrum = apply_phase_weights(spectrum, k)
    
    half_len = len(spectrum) // 2
    magnitudes = np.abs(spectrum[:half_len])
    weighted_magnitudes = np.abs(weighted_spectrum[:half_len])
    
    features = {
        'spectral_disruption_score': compute_spectral_disruption_score(sequence, k, True),
        'baseline_score': compute_spectral_disruption_score(sequence, k, False),
        'mean_magnitude': float(np.mean(magnitudes)),
        'max_magnitude': float(np.max(magnitudes)),
        'dominant_frequency': float(np.argmax(magnitudes)),
        'spectral_entropy': compute_spectral_entropy(magnitudes),
        'weighted_mean_magnitude': float(np.mean(weighted_magnitudes)),
        'delta_magnitude': float(np.mean(weighted_magnitudes) - np.mean(magnitudes)),
        'gc_content': compute_gc_content(sequence),
        'sequence_length': len(sequence)
    }
    
    return features


def compute_spectral_entropy(magnitudes: np.ndarray) -> float:
    """
    Compute spectral entropy from magnitude spectrum.
    
    Args:
        magnitudes: Array of spectral magnitudes
        
    Returns:
        Entropy value
    """
    # Normalize to probability distribution
    magnitudes = magnitudes + 1e-10  # Avoid log(0)
    probs = magnitudes / np.sum(magnitudes)
    
    # Compute Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))
    
    return float(entropy)


def compute_gc_content(sequence: str) -> float:
    """
    Compute GC content of a sequence.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        GC content as a fraction (0 to 1)
    """
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if len(sequence) > 0 else 0.0


def batch_score_sequences(sequences: List[str], 
                          k: float = 0.300,
                          use_phase_weighting: bool = True) -> np.ndarray:
    """
    Compute spectral disruption scores for multiple sequences.
    
    Args:
        sequences: List of DNA sequences
        k: Phase weighting exponent
        use_phase_weighting: Whether to apply phase weighting
        
    Returns:
        Array of scores
    """
    scores = []
    for seq in sequences:
        score = compute_spectral_disruption_score(seq, k, use_phase_weighting)
        scores.append(score)
    
    return np.array(scores)
