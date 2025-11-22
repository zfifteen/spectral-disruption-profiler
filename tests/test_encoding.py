"""
Tests for the encoding module.
"""

import numpy as np
import pytest
from src.encoding import encode_sequence, encode_sequences, pad_sequence


def test_encode_sequence_basic():
    """Test basic DNA sequence encoding."""
    seq = "ATCG"
    encoded = encode_sequence(seq)
    
    expected = np.array([1+0j, -1+0j, 0+1j, 0-1j])
    
    assert len(encoded) == 4
    assert np.allclose(encoded, expected)


def test_encode_sequence_lowercase():
    """Test encoding with lowercase letters."""
    seq = "atcg"
    encoded = encode_sequence(seq)
    
    expected = np.array([1+0j, -1+0j, 0+1j, 0-1j])
    
    assert np.allclose(encoded, expected)


def test_encode_sequence_mixed_case():
    """Test encoding with mixed case."""
    seq = "AtCg"
    encoded = encode_sequence(seq)
    
    expected = np.array([1+0j, -1+0j, 0+1j, 0-1j])
    
    assert np.allclose(encoded, expected)


def test_encode_sequences_multiple():
    """Test encoding multiple sequences."""
    sequences = ["AT", "CG", "ATCG"]
    encoded_list = encode_sequences(sequences)
    
    assert len(encoded_list) == 3
    assert len(encoded_list[0]) == 2
    assert len(encoded_list[1]) == 2
    assert len(encoded_list[2]) == 4


def test_pad_sequence():
    """Test sequence padding."""
    encoded = np.array([1+0j, -1+0j])
    padded = pad_sequence(encoded, 5)
    
    assert len(padded) == 5
    assert np.allclose(padded[:2], encoded)
    assert np.allclose(padded[2:], np.zeros(3, dtype=complex))


def test_pad_sequence_truncate():
    """Test sequence truncation when already longer than target."""
    encoded = np.array([1+0j, -1+0j, 0+1j, 0-1j])
    padded = pad_sequence(encoded, 2)
    
    assert len(padded) == 2
    assert np.allclose(padded, encoded[:2])
