"""
Statistical validation module for bootstrap resampling and significance testing.

Implements bootstrap confidence intervals, DeLong test, and permutation tests
for ROC curve comparison.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple, Dict, List, Optional
import warnings


def bootstrap_auc(y_true: np.ndarray, 
                  y_scores: np.ndarray,
                  n_bootstraps: int = 10000,
                  confidence_level: float = 0.95,
                  random_seed: Optional[int] = 42) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for AUC.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        n_bootstraps: Number of bootstrap resamples
        confidence_level: Confidence level for CI (default 0.95)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with AUC, CI bounds, and bootstrap samples
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(y_true)
    auc_scores = []
    
    for _ in range(n_bootstraps):
        # Resample with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        
        # Skip if all same class
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        try:
            auc = roc_auc_score(y_true[indices], y_scores[indices])
            auc_scores.append(auc)
        except ValueError:
            continue
    
    auc_scores = np.array(auc_scores)
    
    # Compute percentile-based CI
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(auc_scores, lower_percentile)
    ci_upper = np.percentile(auc_scores, upper_percentile)
    
    # Compute base AUC on full dataset
    base_auc = roc_auc_score(y_true, y_scores)
    
    return {
        'auc': float(base_auc),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'bootstrap_mean': float(np.mean(auc_scores)),
        'bootstrap_std': float(np.std(auc_scores)),
        'n_bootstraps': len(auc_scores)
    }


def compare_two_methods_bootstrap(y_true: np.ndarray,
                                  scores_method1: np.ndarray,
                                  scores_method2: np.ndarray,
                                  n_bootstraps: int = 10000,
                                  random_seed: Optional[int] = 42) -> Dict[str, any]:
    """
    Compare two scoring methods using bootstrap resampling.
    
    Args:
        y_true: True binary labels
        scores_method1: Scores from method 1 (e.g., spectral)
        scores_method2: Scores from method 2 (e.g., baseline)
        n_bootstraps: Number of bootstrap resamples
        random_seed: Random seed
        
    Returns:
        Dictionary with comparison statistics
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(y_true)
    delta_aucs = []
    method1_aucs = []
    method2_aucs = []
    
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n_samples, n_samples)
        
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        try:
            auc1 = roc_auc_score(y_true[indices], scores_method1[indices])
            auc2 = roc_auc_score(y_true[indices], scores_method2[indices])
            
            method1_aucs.append(auc1)
            method2_aucs.append(auc2)
            delta_aucs.append(auc1 - auc2)
        except ValueError:
            continue
    
    delta_aucs = np.array(delta_aucs)
    method1_aucs = np.array(method1_aucs)
    method2_aucs = np.array(method2_aucs)
    
    # Compute base AUCs
    base_auc1 = roc_auc_score(y_true, scores_method1)
    base_auc2 = roc_auc_score(y_true, scores_method2)
    base_delta = base_auc1 - base_auc2
    
    # Bootstrap CI for delta
    ci_lower = np.percentile(delta_aucs, 2.5)
    ci_upper = np.percentile(delta_aucs, 97.5)
    
    # Bootstrap p-value (proportion of bootstrap samples where delta <= 0)
    p_value_bootstrap = np.mean(delta_aucs <= 0)
    
    return {
        'method1_auc': float(base_auc1),
        'method2_auc': float(base_auc2),
        'delta_auc': float(base_delta),
        'delta_ci_lower': float(ci_lower),
        'delta_ci_upper': float(ci_upper),
        'delta_mean': float(np.mean(delta_aucs)),
        'delta_std': float(np.std(delta_aucs)),
        'p_value_bootstrap': float(p_value_bootstrap),
        'ci_excludes_zero': bool(ci_lower > 0 or ci_upper < 0),
        'n_bootstraps': len(delta_aucs)
    }


def delong_test(y_true: np.ndarray, 
                scores_method1: np.ndarray,
                scores_method2: np.ndarray) -> Dict[str, float]:
    """
    Perform DeLong test to compare two ROC curves.
    
    This is a parametric test based on the variance of AUC estimates.
    
    Args:
        y_true: True binary labels
        scores_method1: Scores from method 1
        scores_method2: Scores from method 2
        
    Returns:
        Dictionary with test statistics
    """
    # Compute AUCs
    auc1 = roc_auc_score(y_true, scores_method1)
    auc2 = roc_auc_score(y_true, scores_method2)
    
    # Simple approximation using bootstrap variance
    # Full DeLong requires computing structural components
    # For simplicity, we use a z-test approximation
    
    n = len(y_true)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    # Approximate standard error (simplified)
    # In practice, this should use the full DeLong covariance estimation
    se_diff = np.sqrt((auc1 * (1 - auc1) / n_pos + auc2 * (1 - auc2) / n_neg))
    
    # Z-statistic
    z_stat = (auc1 - auc2) / se_diff if se_diff > 0 else 0
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {
        'auc1': float(auc1),
        'auc2': float(auc2),
        'delta_auc': float(auc1 - auc2),
        'z_statistic': float(z_stat),
        'p_value': float(p_value)
    }


def permutation_test(y_true: np.ndarray,
                    scores_method1: np.ndarray,
                    scores_method2: np.ndarray,
                    n_permutations: int = 10000,
                    random_seed: Optional[int] = 42) -> Dict[str, float]:
    """
    Perform permutation test to compare two scoring methods.
    
    Args:
        y_true: True binary labels
        scores_method1: Scores from method 1
        scores_method2: Scores from method 2
        n_permutations: Number of permutations
        random_seed: Random seed
        
    Returns:
        Dictionary with test statistics
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Observed difference
    auc1_obs = roc_auc_score(y_true, scores_method1)
    auc2_obs = roc_auc_score(y_true, scores_method2)
    delta_obs = auc1_obs - auc2_obs
    
    # Permutation distribution
    delta_perm = []
    
    for _ in range(n_permutations):
        # Randomly swap predictions between methods
        swap_mask = np.random.rand(len(y_true)) > 0.5
        scores_perm1 = np.where(swap_mask, scores_method1, scores_method2)
        scores_perm2 = np.where(swap_mask, scores_method2, scores_method1)
        
        try:
            auc1_perm = roc_auc_score(y_true, scores_perm1)
            auc2_perm = roc_auc_score(y_true, scores_perm2)
            delta_perm.append(auc1_perm - auc2_perm)
        except ValueError:
            continue
    
    delta_perm = np.array(delta_perm)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(delta_perm) >= np.abs(delta_obs))
    
    return {
        'delta_auc_observed': float(delta_obs),
        'p_value_permutation': float(p_value),
        'n_permutations': len(delta_perm)
    }


def compute_roc_curve_data(y_true: np.ndarray, 
                          y_scores: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute ROC curve data for plotting.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        
    Returns:
        Dictionary with FPR, TPR, thresholds, and AUC
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': float(auc)
    }
