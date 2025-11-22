"""
Tests for the spectral features module.
"""

import numpy as np
import pytest
from src.spectral_features import (
    compute_fft_spectrum,
    compute_spectral_disruption_score,
    compute_spectral_features,
    compute_gc_content,
    batch_score_sequences
)


def test_compute_fft_spectrum():
    """Test FFT spectrum computation."""
    encoded = np.array([1+0j, -1+0j, 0+1j, 0-1j])
    spectrum = compute_fft_spectrum(encoded)
    
    assert len(spectrum) == len(encoded)
    assert spectrum.dtype == np.complex128


def test_compute_spectral_disruption_score():
    """Test spectral disruption score computation."""
    sequence = "ATCGATCGATCGATCGATCG"
    score = compute_spectral_disruption_score(sequence, k=0.3)
    
    assert isinstance(score, float)
    assert score > 0  # Scores should be positive


def test_compute_spectral_disruption_score_with_without_weighting():
    """Test that phase weighting makes a difference."""
    sequence = "ATCGATCGATCGATCGATCG"
    
    score_with = compute_spectral_disruption_score(sequence, k=0.3, use_phase_weighting=True)
    score_without = compute_spectral_disruption_score(sequence, k=0.3, use_phase_weighting=False)
    
    # Scores should be different
    assert score_with != score_without


def test_compute_gc_content():
    """Test GC content calculation."""
    # 50% GC
    assert np.isclose(compute_gc_content("ATCG"), 0.5)
    
    # 100% GC
    assert np.isclose(compute_gc_content("GCGC"), 1.0)
    
    # 0% GC
    assert np.isclose(compute_gc_content("ATAT"), 0.0)
    
    # 25% GC
    assert np.isclose(compute_gc_content("AAAC"), 0.25)


def test_compute_spectral_features():
    """Test comprehensive feature computation."""
    sequence = "ATCGATCGATCGATCGATCG"
    features = compute_spectral_features(sequence, k=0.3)
    
    # Check that all expected features are present
    expected_keys = [
        'spectral_disruption_score',
        'baseline_score',
        'mean_magnitude',
        'max_magnitude',
        'dominant_frequency',
        'spectral_entropy',
        'weighted_mean_magnitude',
        'delta_magnitude',
        'gc_content',
        'sequence_length'
    ]
    
    for key in expected_keys:
        assert key in features
        assert isinstance(features[key], (int, float))


def test_batch_score_sequences():
    """Test batch scoring of multiple sequences."""
    sequences = [
        "ATCGATCGATCGATCGATCG",
        "AAAATTTTCCCCGGGG",
        "GCGCGCGCGCGCGCGCGCGC"
    ]
    
    scores = batch_score_sequences(sequences, k=0.3, use_phase_weighting=True)
    
    assert len(scores) == len(sequences)
    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.float64
    assert np.all(scores > 0)  # All scores should be positive


def test_different_sequences_different_scores():
    """Test that different sequences get different scores."""
    seq1 = "AAAAAAAAAAAAAAAAAAAA"  # All A
    seq2 = "GCGCGCGCGCGCGCGCGCGC"  # Alternating G/C
    
    score1 = compute_spectral_disruption_score(seq1, k=0.3)
    score2 = compute_spectral_disruption_score(seq2, k=0.3)
    
    # These should have different spectral properties
    assert score1 != score2
