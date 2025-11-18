"""
Tests for the dataset loader module.
"""

import numpy as np
import pytest
import pandas as pd
from src.dataset_loader import (
    generate_synthetic_dataset,
    prepare_dataset_for_validation,
    create_train_test_split
)


def test_generate_synthetic_dataset_shape():
    """Test that synthetic dataset has correct shape."""
    n_samples = 100
    df = generate_synthetic_dataset(n_samples=n_samples, random_seed=42)
    
    assert len(df) == n_samples
    assert 'sequence' in df.columns
    assert 'efficacy' in df.columns
    assert 'gc_content' in df.columns


def test_generate_synthetic_dataset_reproducibility():
    """Test that synthetic dataset generation is reproducible."""
    df1 = generate_synthetic_dataset(n_samples=100, random_seed=42)
    df2 = generate_synthetic_dataset(n_samples=100, random_seed=42)
    
    assert df1.equals(df2)


def test_generate_synthetic_dataset_efficacy_range():
    """Test that efficacy scores are in valid range."""
    df = generate_synthetic_dataset(n_samples=100, random_seed=42)
    
    assert df['efficacy'].min() >= 0
    assert df['efficacy'].max() <= 1


def test_generate_synthetic_dataset_sequence_length():
    """Test that generated sequences have correct length."""
    df = generate_synthetic_dataset(n_samples=100, random_seed=42)
    
    # All sequences should be 20-mers
    sequence_lengths = df['sequence'].apply(len)
    assert all(sequence_lengths == 20)


def test_generate_synthetic_dataset_gc_variation():
    """Test that GC content varies across dataset."""
    df = generate_synthetic_dataset(n_samples=1000, random_seed=42)
    
    gc_counts = df['gc_content'].values / 20.0  # Convert to fraction
    
    # Should have a range of GC contents
    assert gc_counts.min() < 0.3
    assert gc_counts.max() > 0.7


def test_prepare_dataset_for_validation():
    """Test dataset preparation for validation."""
    df = generate_synthetic_dataset(n_samples=100, random_seed=42)
    
    sequences, labels = prepare_dataset_for_validation(df, threshold=0.5)
    
    assert len(sequences) == len(df)
    assert len(labels) == len(df)
    assert set(labels) == {0, 1} or set(labels) == {0} or set(labels) == {1}


def test_prepare_dataset_threshold_effect():
    """Test that threshold affects binarization."""
    df = generate_synthetic_dataset(n_samples=100, random_seed=42)
    
    _, labels_low = prepare_dataset_for_validation(df, threshold=0.3)
    _, labels_high = prepare_dataset_for_validation(df, threshold=0.7)
    
    # With lower threshold, should have more high-efficacy (1s)
    assert np.sum(labels_low) >= np.sum(labels_high)


def test_create_train_test_split():
    """Test train/test split."""
    n_samples = 100
    sequences = np.array(['SEQ' + str(i) for i in range(n_samples)])
    labels = np.random.randint(0, 2, n_samples)
    
    train_seq, test_seq, train_labels, test_labels = create_train_test_split(
        sequences, labels, test_size=0.2, random_seed=42
    )
    
    # Check sizes
    assert len(test_seq) == 20
    assert len(train_seq) == 80
    assert len(test_labels) == 20
    assert len(train_labels) == 80
    
    # Check no overlap
    assert len(set(train_seq) & set(test_seq)) == 0


def test_create_train_test_split_reproducibility():
    """Test that train/test split is reproducible."""
    sequences = np.array(['SEQ' + str(i) for i in range(100)])
    labels = np.random.randint(0, 2, 100)
    
    result1 = create_train_test_split(sequences, labels, random_seed=42)
    result2 = create_train_test_split(sequences, labels, random_seed=42)
    
    # Check that splits are identical
    assert np.array_equal(result1[0], result2[0])  # train_sequences
    assert np.array_equal(result1[1], result2[1])  # test_sequences
    assert np.array_equal(result1[2], result2[2])  # train_labels
    assert np.array_equal(result1[3], result2[3])  # test_labels
