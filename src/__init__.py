"""Spectral Disruption Profiler package."""

from .encoding import encode_sequence, encode_sequences, pad_sequence
from .phase_weighting import compute_phase_weights, apply_phase_weights, sweep_k_values, PHI
from .spectral_features import (
    compute_fft_spectrum,
    compute_spectral_disruption_score,
    compute_spectral_features,
    batch_score_sequences
)
from .statistical_validation import (
    bootstrap_auc,
    compare_two_methods_bootstrap,
    delong_test,
    permutation_test,
    compute_roc_curve_data
)
from .dataset_loader import (
    generate_synthetic_dataset,
    prepare_dataset_for_validation,
    load_dataset_by_name
)

__all__ = [
    'encode_sequence',
    'encode_sequences',
    'pad_sequence',
    'compute_phase_weights',
    'apply_phase_weights',
    'sweep_k_values',
    'PHI',
    'compute_fft_spectrum',
    'compute_spectral_disruption_score',
    'compute_spectral_features',
    'batch_score_sequences',
    'bootstrap_auc',
    'compare_two_methods_bootstrap',
    'delong_test',
    'permutation_test',
    'compute_roc_curve_data',
    'generate_synthetic_dataset',
    'prepare_dataset_for_validation',
    'load_dataset_by_name'
]
