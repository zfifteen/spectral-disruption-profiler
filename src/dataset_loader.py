"""
Dataset loader and management module.

Handles loading and preparing public gRNA efficacy datasets.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import os


def generate_synthetic_dataset(n_samples: int = 1000, 
                               random_seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Generate a synthetic gRNA dataset for testing and demonstration.
    
    This creates sequences with varying GC content and efficacy scores
    that correlate with spectral properties.
    
    Args:
        n_samples: Number of sequences to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: sequence, efficacy, gc_content
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    sequences = []
    efficacy_scores = []
    
    bases = ['A', 'T', 'C', 'G']
    
    for i in range(n_samples):
        # Generate random 20-mer gRNA sequence (typical length)
        length = 20
        
        # Vary GC content across dataset (20% to 80%)
        gc_target = 0.2 + 0.6 * (i / n_samples)
        
        sequence = []
        for _ in range(length):
            if np.random.rand() < gc_target:
                # Use G or C
                sequence.append(np.random.choice(['G', 'C']))
            else:
                # Use A or T
                sequence.append(np.random.choice(['A', 'T']))
        
        seq_str = ''.join(sequence)
        sequences.append(seq_str)
        
        # Generate efficacy score based on GC content and spectral properties
        # Higher GC content (near 50%) tends to be better
        gc_actual = (seq_str.count('G') + seq_str.count('C')) / length
        
        # Optimal GC around 0.5, with noise
        efficacy_base = 1.0 - 2.0 * abs(gc_actual - 0.5)
        
        # Add spectral-like feature: penalize very repetitive sequences
        max_repeat = max([seq_str.count(b * 3) for b in bases])
        repeat_penalty = max_repeat * 0.1
        
        # Add noise
        noise = np.random.normal(0, 0.15)
        
        efficacy = np.clip(efficacy_base - repeat_penalty + noise, 0, 1)
        efficacy_scores.append(efficacy)
    
    df = pd.DataFrame({
        'sequence': sequences,
        'efficacy': efficacy_scores,
        'gc_content': [s.count('G') + s.count('C') for s in sequences]
    })
    
    return df


def prepare_dataset_for_validation(df: pd.DataFrame,
                                   label_column: str = 'efficacy',
                                   sequence_column: str = 'sequence',
                                   threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare a dataset for validation by converting to binary labels.
    
    Args:
        df: DataFrame with sequences and efficacy scores
        label_column: Name of the efficacy score column
        sequence_column: Name of the sequence column
        threshold: Threshold for binarization (high vs low efficacy)
        
    Returns:
        Tuple of (sequences as list, binary labels as array)
    """
    sequences = df[sequence_column].values
    efficacy = df[label_column].values
    
    # Binarize: 1 for high efficacy, 0 for low
    binary_labels = (efficacy >= threshold).astype(int)
    
    return sequences, binary_labels


def load_doench_2016_dataset() -> Optional[pd.DataFrame]:
    """
    Load Doench 2016 dataset if available.
    
    This is a placeholder - in practice, this would download or load
    the actual Doench 2016 data from a public source.
    
    Returns:
        DataFrame or None if not available
    """
    # TODO: Implement actual data loading
    # For now, return None to indicate it's not available
    # In a real implementation, this would fetch from:
    # - Supplementary data from the publication
    # - A public repository like Zenodo
    # - Or generate from raw data files
    
    return None


def load_dataset_by_name(dataset_name: str) -> Optional[pd.DataFrame]:
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'doench2016', 'synthetic')
        
    Returns:
        DataFrame or None if not available
    """
    if dataset_name.lower() == 'synthetic':
        return generate_synthetic_dataset()
    elif dataset_name.lower() == 'doench2016':
        return load_doench_2016_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_train_test_split(sequences: np.ndarray,
                           labels: np.ndarray,
                           test_size: float = 0.2,
                           random_seed: Optional[int] = 42) -> Tuple:
    """
    Create train/test split for validation.
    
    Args:
        sequences: Array of sequences
        labels: Array of labels
        test_size: Fraction for test set
        random_seed: Random seed
        
    Returns:
        Tuple of (train_sequences, test_sequences, train_labels, test_labels)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(sequences)
    n_test = int(n_samples * test_size)
    
    # Random permutation
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return (
        sequences[train_indices],
        sequences[test_indices],
        labels[train_indices],
        labels[test_indices]
    )
