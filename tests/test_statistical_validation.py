"""
Tests for the statistical validation module.
"""

import numpy as np
import pytest
from src.statistical_validation import (
    bootstrap_auc,
    compare_two_methods_bootstrap,
    delong_test,
    permutation_test,
    compute_roc_curve_data
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 100
    
    # Binary labels
    y_true = np.random.randint(0, 2, n)
    
    # Two sets of scores with some correlation to truth
    y_scores1 = y_true + np.random.randn(n) * 0.5
    y_scores2 = y_true + np.random.randn(n) * 0.7
    
    return y_true, y_scores1, y_scores2


def test_bootstrap_auc(sample_data):
    """Test bootstrap AUC computation."""
    y_true, y_scores, _ = sample_data
    
    result = bootstrap_auc(y_true, y_scores, n_bootstraps=100, random_seed=42)
    
    # Check that all expected keys are present
    assert 'auc' in result
    assert 'ci_lower' in result
    assert 'ci_upper' in result
    assert 'bootstrap_mean' in result
    assert 'bootstrap_std' in result
    assert 'n_bootstraps' in result
    
    # Check ranges
    assert 0 <= result['auc'] <= 1
    assert 0 <= result['ci_lower'] <= result['ci_upper'] <= 1
    assert result['bootstrap_std'] >= 0


def test_bootstrap_reproducibility(sample_data):
    """Test that bootstrap results are reproducible with same seed."""
    y_true, y_scores, _ = sample_data
    
    result1 = bootstrap_auc(y_true, y_scores, n_bootstraps=100, random_seed=42)
    result2 = bootstrap_auc(y_true, y_scores, n_bootstraps=100, random_seed=42)
    
    assert result1['auc'] == result2['auc']
    assert result1['ci_lower'] == result2['ci_lower']
    assert result1['ci_upper'] == result2['ci_upper']


def test_compare_two_methods_bootstrap(sample_data):
    """Test comparison of two methods using bootstrap."""
    y_true, scores1, scores2 = sample_data
    
    result = compare_two_methods_bootstrap(
        y_true, scores1, scores2,
        n_bootstraps=100,
        random_seed=42
    )
    
    # Check expected keys
    assert 'method1_auc' in result
    assert 'method2_auc' in result
    assert 'delta_auc' in result
    assert 'delta_ci_lower' in result
    assert 'delta_ci_upper' in result
    assert 'p_value_bootstrap' in result
    assert 'ci_excludes_zero' in result
    
    # Check that delta is difference
    assert np.isclose(
        result['delta_auc'],
        result['method1_auc'] - result['method2_auc']
    )
    
    # Check p-value range
    assert 0 <= result['p_value_bootstrap'] <= 1


def test_delong_test(sample_data):
    """Test DeLong test for ROC comparison."""
    y_true, scores1, scores2 = sample_data
    
    result = delong_test(y_true, scores1, scores2)
    
    # Check expected keys
    assert 'auc1' in result
    assert 'auc2' in result
    assert 'delta_auc' in result
    assert 'z_statistic' in result
    assert 'p_value' in result
    
    # Check ranges
    assert 0 <= result['auc1'] <= 1
    assert 0 <= result['auc2'] <= 1
    assert 0 <= result['p_value'] <= 1


def test_permutation_test(sample_data):
    """Test permutation test."""
    y_true, scores1, scores2 = sample_data
    
    result = permutation_test(
        y_true, scores1, scores2,
        n_permutations=100,
        random_seed=42
    )
    
    # Check expected keys
    assert 'delta_auc_observed' in result
    assert 'p_value_permutation' in result
    assert 'n_permutations' in result
    
    # Check p-value range
    assert 0 <= result['p_value_permutation'] <= 1


def test_compute_roc_curve_data(sample_data):
    """Test ROC curve computation."""
    y_true, y_scores, _ = sample_data
    
    result = compute_roc_curve_data(y_true, y_scores)
    
    # Check expected keys
    assert 'fpr' in result
    assert 'tpr' in result
    assert 'thresholds' in result
    assert 'auc' in result
    
    # Check shapes
    assert len(result['fpr']) == len(result['tpr'])
    assert len(result['fpr']) == len(result['thresholds'])
    
    # Check ranges
    assert np.all(result['fpr'] >= 0) and np.all(result['fpr'] <= 1)
    assert np.all(result['tpr'] >= 0) and np.all(result['tpr'] <= 1)
    assert 0 <= result['auc'] <= 1
    
    # Check that curve starts at (0,0) or close to it
    assert result['fpr'][0] <= 0.1 or result['tpr'][0] <= 0.1
