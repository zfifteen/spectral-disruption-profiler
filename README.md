# Spectral Disruption Profiler

A cloud-based SaaS platform for CRISPR gRNA optimization using Fourier-based signal processing on DNA sequences to quantify mutation-induced disruptions.

## Overview

This platform applies FFT-encoded DNA waveform analysis with θ′(n,k) phase weighting (optimal k* ≈ 0.300) to score gRNA on-target efficiency and off-target risks using Z-invariant mutation metrics.

## Key Features

- **Sequence Upload and Encoding**: Support bulk upload of gRNA sequences (FASTA/CSV formats) with automatic complex waveform encoding and θ′(n,k) phase application.
- **Spectral Analysis Engine**: Compute FFT-based features, including disruption deltas (Δf₁), entropy gradients (ΔEntropy), and sidelobe metrics, biased by golden-ratio spirals for low-discrepancy sampling.
- **Z-Invariant Scoring**: Generate composite scores invariant to sequence transformations, with bootstrap CI reporting for confidence.
- **Off-Target Detection**: Flag risks via entropy gradients and GC-quartile resonance, comparing to benchmarks like Doench 2016 or Patch 2024.
- **Visualization and Reporting**: Interactive dashboards with waveform plots, ROC curves, and ranked variant lists; export to CSV/JSON for downstream ML integration.
- **API Integration**: RESTful endpoints for programmatic access, e.g., scoring via POST requests with sequence payloads.

## Acceptance Criteria

- **Functional**: Upload a batch of 100+ gRNA sequences processes in <30s, outputting scores with phase-weighted adjustments matching repo benchmarks (+0.047 ΔROC-AUC vs. RuleSet3).
- **Performance**: 99% uptime for cloud SaaS; handle 1,000 queries/day per user tier; bootstrap CI computation (4,000–10,000 resamples) completes in <5s per sequence.
- **Usability**: Intuitive UI with tooltips explaining θ′(n,k) and Z-invariants; no prior math expertise required; error handling for invalid sequences.
- **Security/Compliance**: HIPAA/GDPR compliant data handling; encrypted storage for proprietary sequences; audit logs for all analyses.
- **Empirical Validation**: Integrated demo mode using synthetic data from proof_pack/generate_synthetic_data.py, reproducing GC-bias tests (variable GC 20–80%); A/B testing comparing platform scores to manual RuleSet3 on user-provided holdout sets, requiring >0% lift in prediction accuracy.

## User Story

As a biopharma researcher specializing in gene editing, I want a cloud-based SaaS platform that applies FFT-encoded DNA waveform analysis with θ′(n,k) phase weighting (optimal k* ≈ 0.300) to score gRNA on-target efficiency and off-target risks using Z-invariant mutation metrics, so that I can rank and select high-performing guide RNA variants more accurately, reducing experimental failures and accelerating therapeutic development in the CRISPR gene editing market, projected at USD 7.06 billion in 2025.

## Background and Rationale

This derives from the wave-crispr-signal framework, encoding DNA as complex waveforms (A→1+0i, T→-1+0i, C→0+1i, G→0-1i) and using FFT to quantify spectral disruption from mutations. Phase weighting via θ′(n,k) = φ · ((n mod φ)/φ)^k incorporates golden-ratio geometry for enhanced resolution, linking to Z Framework invariants for robust scoring. Empirical validation on datasets like Kim 2025 (N=18,102 guides) shows ΔROC-AUC of +0.047 ± 0.006 (10,000× bootstrap) over baselines like RuleSet3, with GC-resonance correlations (r = –0.211 in Q4, p_perm = 0.0012, FDR-corrected) highlighting off-target entropy gradients.

## Risks and Mitigations

- **Risk**: Over-reliance on k* ≈ 0.300 without dataset-specific tuning. Mitigation: Provide auto-optimization mode using geodesic-topological bridge validation, with fallback to default.
- **Risk**: Computational intensity for large N (>10^5). Mitigation: Batch processing with vectorized FFT (NumPy/SciPy); cloud scaling via AWS/GCP.
- **Risk**: False positives in off-target detection. Mitigation: FDR-correction in p-value reporting; user-configurable thresholds.

## Next Steps for Implementation

- Prototype using repo components: z_framework.py for core, spectral_features.py for FFT, topological_analysis.py for bridges.
- Benchmark on notebooks/compare_ruleset3_wave.ipynb; log results to CSV with CI.
- Deploy as SaaS MVP: Flask/Django backend, React frontend; integrate CLI from applications/phase_weighted_scorecard_cli.py.
- Test Plan: Run proof_pack/run_validation.py equivalents; target 70% coverage.

## Monetization

Freemium tier (limited queries), premium for unlimited + API ($99/mo).

## Project Structure

- `docs/`: Documentation
- `src/`: Source code (to be added)
- `tests/`: Tests (to be added)
- `requirements.txt`: Dependencies (to be added)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/zfifteen/spectral-disruption-profiler.git
cd spectral-disruption-profiler
pip install -r requirements.txt
```

## Usage

(TBD - API and UI usage)

## Contributing

Please see CONTRIBUTING.md for details.

## License

MIT License.