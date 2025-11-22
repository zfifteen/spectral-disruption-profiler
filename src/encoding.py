"""
DNA sequence encoding module for spectral analysis.

Encodes DNA sequences as complex waveforms for FFT-based analysis:
- A → 1+0i
- T → -1+0i
- C → 0+1i
- G → 0-1i
"""

import numpy as np
from typing import List, Union


def encode_sequence(sequence: str) -> np.ndarray:
    """
    Encode a DNA sequence as a complex waveform.
    
    Args:
        sequence: DNA sequence string (A, T, C, G)
        
    Returns:
        Complex numpy array representing the encoded sequence
    """
    encoding_map = {
        'A': 1 + 0j,
        'T': -1 + 0j,
        'C': 0 + 1j,
        'G': 0 - 1j,
        'a': 1 + 0j,
        't': -1 + 0j,
        'c': 0 + 1j,
        'g': 0 - 1j
    }
    
    # Convert sequence to uppercase and encode
    return np.array([encoding_map.get(base, 0+0j) for base in sequence], dtype=complex)


def encode_sequences(sequences: List[str]) -> List[np.ndarray]:
    """
    Encode multiple DNA sequences as complex waveforms.
    
    Args:
        sequences: List of DNA sequence strings
        
    Returns:
        List of complex numpy arrays
    """
    return [encode_sequence(seq) for seq in sequences]


def pad_sequence(encoded: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad an encoded sequence to a target length with zeros.
    
    Args:
        encoded: Complex encoded sequence
        target_length: Desired length
        
    Returns:
        Padded sequence
    """
    if len(encoded) >= target_length:
        return encoded[:target_length]
    
    padding = np.zeros(target_length - len(encoded), dtype=complex)
    return np.concatenate([encoded, padding])
