"""
Main validation pipeline for spectral disruption profiler.

This script runs the complete validation pipeline including:
1. Loading datasets
2. Computing spectral disruption scores
3. Bootstrap resampling (10,000 iterations)
4. Statistical significance testing
5. k-sweep analysis
6. Generating output artifacts

Usage:
    python run_validation.py --dataset synthetic --seed 42
    python run_validation.py --all-datasets --seed 42
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.spectral_features import batch_score_sequences, compute_spectral_features
from src.statistical_validation import (
    bootstrap_auc, 
    compare_two_methods_bootstrap,
    delong_test,
    permutation_test,
    compute_roc_curve_data
)
from src.dataset_loader import (
    load_dataset_by_name,
    prepare_dataset_for_validation,
    generate_synthetic_dataset
)
from src.phase_weighting import sweep_k_values


def validate_dataset(dataset_name: str,
                     k_value: float = 0.300,
                     n_bootstraps: int = 10000,
                     random_seed: int = 42,
                     output_dir: str = 'results') -> Dict:
    """
    Run complete validation pipeline on a single dataset.
    
    Args:
        dataset_name: Name of dataset to validate
        k_value: Phase weighting parameter k
        n_bootstraps: Number of bootstrap resamples
        random_seed: Random seed for reproducibility
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all validation results
    """
    print(f"\n{'='*80}")
    print(f"Validating dataset: {dataset_name}")
    print(f"k-value: {k_value}, n_bootstraps: {n_bootstraps}, seed: {random_seed}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir = Path(output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}...")
    df = load_dataset_by_name(dataset_name)
    
    if df is None:
        print(f"WARNING: Dataset {dataset_name} not available. Skipping.")
        return None
    
    print(f"  Loaded {len(df)} sequences")
    
    # Prepare for validation
    sequences, labels = prepare_dataset_for_validation(df)
    print(f"  Binary labels: {np.sum(labels)} high-efficacy, {len(labels) - np.sum(labels)} low-efficacy")
    
    # Compute spectral scores with phase weighting
    print(f"\nComputing spectral disruption scores (k={k_value})...")
    spectral_scores = batch_score_sequences(sequences, k=k_value, use_phase_weighting=True)
    
    # Compute baseline scores (no phase weighting)
    print(f"Computing baseline scores (no phase weighting)...")
    baseline_scores = batch_score_sequences(sequences, k=k_value, use_phase_weighting=False)
    
    # Save raw predictions
    print(f"\nSaving raw predictions...")
    predictions_df = pd.DataFrame({
        'sequence': sequences,
        'ground_truth': labels,
        'spectral_score': spectral_scores,
        'baseline_score': baseline_scores,
        'efficacy': df['efficacy'].values
    })
    predictions_file = output_dir / f'{dataset_name}_predictions.csv'
    predictions_df.to_csv(predictions_file, index=False)
    print(f"  Saved to: {predictions_file}")
    
    # Bootstrap validation for spectral method
    print(f"\nRunning bootstrap validation (n={n_bootstraps})...")
    print(f"  Spectral method...")
    bootstrap_spectral = bootstrap_auc(
        labels, spectral_scores, 
        n_bootstraps=n_bootstraps,
        random_seed=random_seed
    )
    
    print(f"  Baseline method...")
    bootstrap_baseline = bootstrap_auc(
        labels, baseline_scores,
        n_bootstraps=n_bootstraps,
        random_seed=random_seed + 1
    )
    
    # Compare methods
    print(f"\nComparing methods...")
    comparison = compare_two_methods_bootstrap(
        labels, spectral_scores, baseline_scores,
        n_bootstraps=n_bootstraps,
        random_seed=random_seed + 2
    )
    
    # DeLong test
    print(f"Running DeLong test...")
    delong_result = delong_test(labels, spectral_scores, baseline_scores)
    
    # Permutation test
    print(f"Running permutation test...")
    perm_result = permutation_test(
        labels, spectral_scores, baseline_scores,
        n_permutations=10000,
        random_seed=random_seed + 3
    )
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS FOR {dataset_name}")
    print(f"{'='*80}")
    print(f"\nSpectral Method (k={k_value}):")
    print(f"  AUC: {bootstrap_spectral['auc']:.4f}")
    print(f"  95% CI: [{bootstrap_spectral['ci_lower']:.4f}, {bootstrap_spectral['ci_upper']:.4f}]")
    print(f"\nBaseline Method:")
    print(f"  AUC: {bootstrap_baseline['auc']:.4f}")
    print(f"  95% CI: [{bootstrap_baseline['ci_lower']:.4f}, {bootstrap_baseline['ci_upper']:.4f}]")
    print(f"\nComparison (Spectral vs Baseline):")
    print(f"  ΔROC-AUC: {comparison['delta_auc']:.4f} ± {comparison['delta_std']:.4f}")
    print(f"  95% CI: [{comparison['delta_ci_lower']:.4f}, {comparison['delta_ci_upper']:.4f}]")
    print(f"  CI excludes zero: {comparison['ci_excludes_zero']}")
    print(f"  Bootstrap p-value: {comparison['p_value_bootstrap']:.6f}")
    print(f"  DeLong p-value: {delong_result['p_value']:.6f}")
    print(f"  Permutation p-value: {perm_result['p_value_permutation']:.6f}")
    print(f"{'='*80}\n")
    
    # Save bootstrap artifacts
    bootstrap_file = output_dir / f'{dataset_name}_bootstrap_results.json'
    with open(bootstrap_file, 'w') as f:
        results = {
            'dataset': dataset_name,
            'k_value': k_value,
            'n_bootstraps': n_bootstraps,
            'random_seed': random_seed,
            'spectral': bootstrap_spectral,
            'baseline': bootstrap_baseline,
            'comparison': comparison,
            'delong_test': delong_result,
            'permutation_test': perm_result,
            'timestamp': datetime.now().isoformat()
        }
        json.dump(results, f, indent=2)
    print(f"Bootstrap results saved to: {bootstrap_file}")
    
    # Plot ROC curves
    print(f"\nGenerating ROC curves...")
    roc_spectral = compute_roc_curve_data(labels, spectral_scores)
    roc_baseline = compute_roc_curve_data(labels, baseline_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(roc_spectral['fpr'], roc_spectral['tpr'], 
             label=f'Spectral (AUC={roc_spectral["auc"]:.3f})', linewidth=2)
    plt.plot(roc_baseline['fpr'], roc_baseline['tpr'],
             label=f'Baseline (AUC={roc_baseline["auc"]:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {dataset_name}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    roc_file = output_dir / f'{dataset_name}_roc_curve.png'
    plt.savefig(roc_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {roc_file}")
    
    return results


def k_sweep_analysis(dataset_name: str,
                    k_min: float = 0.20,
                    k_max: float = 0.40,
                    k_step: float = 0.025,
                    random_seed: int = 42,
                    output_dir: str = 'results') -> Dict:
    """
    Perform k-sweep analysis to find optimal k*.
    
    Args:
        dataset_name: Name of dataset
        k_min: Minimum k value
        k_max: Maximum k value
        k_step: Step size
        random_seed: Random seed
        output_dir: Output directory
        
    Returns:
        Dictionary with k-sweep results
    """
    print(f"\n{'='*80}")
    print(f"K-SWEEP ANALYSIS: {dataset_name}")
    print(f"k range: [{k_min}, {k_max}], step: {k_step}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir = Path(output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    df = load_dataset_by_name(dataset_name)
    if df is None:
        print(f"WARNING: Dataset {dataset_name} not available. Skipping k-sweep.")
        return None
    
    sequences, labels = prepare_dataset_for_validation(df)
    
    # Sweep k values
    k_values = np.arange(k_min, k_max + k_step, k_step)
    results = []
    
    for k in k_values:
        print(f"Testing k={k:.3f}...")
        scores = batch_score_sequences(sequences, k=k, use_phase_weighting=True)
        
        # Compute AUC with bootstrap CI
        bootstrap_result = bootstrap_auc(
            labels, scores,
            n_bootstraps=1000,  # Fewer for speed
            random_seed=random_seed
        )
        
        results.append({
            'k': float(k),
            'auc': bootstrap_result['auc'],
            'ci_lower': bootstrap_result['ci_lower'],
            'ci_upper': bootstrap_result['ci_upper']
        })
        
        print(f"  AUC: {bootstrap_result['auc']:.4f} "
              f"[{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")
    
    # Find optimal k
    aucs = [r['auc'] for r in results]
    optimal_idx = np.argmax(aucs)
    optimal_k = results[optimal_idx]['k']
    
    print(f"\nOptimal k*: {optimal_k:.3f}")
    print(f"Maximum AUC: {results[optimal_idx]['auc']:.4f}")
    
    # Save results
    k_sweep_df = pd.DataFrame(results)
    k_sweep_file = output_dir / f'{dataset_name}_k_sweep.csv'
    k_sweep_df.to_csv(k_sweep_file, index=False)
    print(f"K-sweep results saved to: {k_sweep_file}")
    
    # Plot k-sweep
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        k_sweep_df['k'], 
        k_sweep_df['auc'],
        yerr=[
            k_sweep_df['auc'] - k_sweep_df['ci_lower'],
            k_sweep_df['ci_upper'] - k_sweep_df['auc']
        ],
        marker='o', capsize=5, linewidth=2, markersize=6
    )
    plt.axvline(x=0.300, color='r', linestyle='--', label='k*=0.300 (hypothesis)', linewidth=2)
    plt.axvline(x=optimal_k, color='g', linestyle='--', label=f'k*={optimal_k:.3f} (optimal)', linewidth=2)
    plt.xlabel('k value', fontsize=12)
    plt.ylabel('AUC (with 95% CI)', fontsize=12)
    plt.title(f'K-Sweep Analysis - {dataset_name}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    k_sweep_plot = output_dir / f'{dataset_name}_k_sweep.png'
    plt.savefig(k_sweep_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"K-sweep plot saved to: {k_sweep_plot}")
    
    return {
        'dataset': dataset_name,
        'optimal_k': optimal_k,
        'optimal_auc': results[optimal_idx]['auc'],
        'k_sweep_results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run validation pipeline for spectral disruption profiler'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='synthetic',
        help='Dataset to validate (default: synthetic)'
    )
    parser.add_argument(
        '--all-datasets',
        action='store_true',
        help='Run validation on all available datasets'
    )
    parser.add_argument(
        '--k',
        type=float,
        default=0.300,
        help='Phase weighting parameter k (default: 0.300)'
    )
    parser.add_argument(
        '--n-bootstraps',
        type=int,
        default=10000,
        help='Number of bootstrap resamples (default: 10000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    parser.add_argument(
        '--k-sweep',
        action='store_true',
        help='Perform k-sweep analysis'
    )
    
    args = parser.parse_args()
    
    # Set random seed globally
    np.random.seed(args.seed)
    
    print(f"\n{'='*80}")
    print("SPECTRAL DISRUPTION PROFILER - VALIDATION PIPELINE")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Determine which datasets to run
    if args.all_datasets:
        datasets = ['synthetic']  # Add more as they become available
    else:
        datasets = [args.dataset]
    
    # Run validation on each dataset
    all_results = []
    for dataset in datasets:
        result = validate_dataset(
            dataset,
            k_value=args.k,
            n_bootstraps=args.n_bootstraps,
            random_seed=args.seed,
            output_dir=args.output_dir
        )
        
        if result is not None:
            all_results.append(result)
        
        # K-sweep if requested
        if args.k_sweep:
            k_sweep_result = k_sweep_analysis(
                dataset,
                random_seed=args.seed,
                output_dir=args.output_dir
            )
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Validated {len(all_results)} dataset(s)")
    print(f"Results saved to: {args.output_dir}/")
    print(f"{'='*80}\n")
    
    # Create summary file
    summary_file = Path(args.output_dir) / 'validation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'k_value': args.k,
                'n_bootstraps': args.n_bootstraps,
                'random_seed': args.seed
            },
            'datasets': all_results
        }, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
