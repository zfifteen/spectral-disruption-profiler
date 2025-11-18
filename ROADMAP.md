# Roadmap: Spectral Disruption Profiler

## Premise

FFT-encoded DNA with θ′(n,k) phase weighting (k* ≈ 0.300) produces measurably better gRNA scoring than RuleSet3. Kim 2025 showed +0.047 ± 0.006 ΔROC-AUC. The question: does this generalize, or is it dataset-specific noise?

Prove the invariant holds across multiple independent datasets before building anything else. No SaaS infrastructure, no pricing tiers, no user stories until the algorithm demonstrates universal signal.

---

## Phase 1: Core Validation (8 weeks)

**Goal:** Prove k* ≈ 0.300 generalizes across ≥3 independent gRNA efficacy datasets with ΔROC-AUC ≥ 0.03, p < 0.01 per dataset.

### Deliverables

1. **CLI scoring tool** (`score_grna.py`)
   - Input: FASTA + measured efficacy CSV
   - Output: scored predictions, bootstrap CI (10,000 resamples), ROC curves, raw artifacts (CSV/JSON)
   - Logged: k value, precision (mp.dps), FFT params, dataset provenance, timestamp
   - Dependencies: NumPy/SciPy for FFT, mpmath for precision, scikit-learn for metrics

2. **Multi-dataset validation**
   - **Required datasets:**
     - Kim 2025 (N=18,102) — baseline, already tested
     - Doench 2016 (N=~5,000) — industry standard
     - One additional: Patch 2024, Wang 2019, or equivalent (N>1,000)
   - **Success gate:** ΔROC-AUC ≥ 0.03 with bootstrap 95% CI excluding zero on all three
   - **Failure case:** If any dataset shows ΔROC-AUC < 0.02 or CI includes zero, investigate k drift

3. **k-optimization sweep**
   - Test k ∈ [0.200, 0.400] in 0.025 increments per dataset
   - Record optimal k* per dataset with CI
   - If k* variance across datasets exceeds ±0.05, document dataset-specific tuning protocol
   - Otherwise, confirm k* = 0.300 as universal default

4. **Reproducibility artifacts**
   - Version all datasets (SHA-256 hash logged)
   - Pin random seeds for bootstrap
   - Export full results: predictions, scores, CIs, plots, parameters
   - README with exact commands to reproduce each validation run

### Gates

- ✅ CLI tool runs end-to-end on Kim 2025, reproduces +0.047 ± 0.006
- ✅ Doench 2016 validation: ΔROC-AUC ≥ 0.03, p < 0.01
- ✅ Third dataset validation: same threshold
- ✅ All artifacts exportable and reproducible from logged parameters

### Anti-goals

- ❌ No web UI
- ❌ No cloud deployment
- ❌ No authentication/encryption/compliance infrastructure
- ❌ No "auto-optimization" unless k* demonstrably varies >0.05 across datasets

---

## Phase 2: Package and Document (4 weeks)

**Prerequisite:** Phase 1 gates passed on ≥3 datasets.

**Goal:** Create a minimal, reproducible package that others can use and verify.

### Deliverables

1. **Python package** (`spectral_disruption`)
   - Core modules: `encoding.py`, `phase_weighting.py`, `spectral_features.py`, `scoring.py`
   - Single entry point: `spectral_disruption.score(sequences, efficacy, k=0.300)`
   - Returns: DataFrame with scores, CIs, feature vectors
   - Dependencies minimal: NumPy, SciPy, mpmath, pandas

2. **Test suite**
   - Unit tests: encoding, FFT transform, θ′ phase application, score computation
   - Integration test: reproduce Kim 2025 result ± tolerance
   - Target: 80% coverage on core scoring path
   - CI: GitHub Actions, runs on push

3. **Documentation**
   - Theory brief: θ′(n,k) formula, Z-invariant connection, why golden ratio
   - Usage: install, run, interpret output
   - Validation results: table with all datasets, ΔROC-AUC, CIs, p-values
   - Provenance: dataset sources, versions, benchmark comparisons
   - Constraints: validated on N ∈ [1,000, 20,000]; assumes FASTA + efficacy labels

4. **Benchmark comparison script**
   - Compare to RuleSet3, DeepHF (if accessible), or other published baselines
   - Output: side-by-side ROC curves, statistical tests (DeLong, permutation)
   - Logged artifacts for each comparison

### Gates

- ✅ Package installable via pip/conda
- ✅ Test suite passes with 80%+ coverage
- ✅ Documentation sufficient for independent reproduction
- ✅ Benchmark script reproduces Phase 1 results on all datasets

### Anti-goals

- ❌ No feature creep: off-target detection, entropy gradients, GC-bias analysis only if they pass same validation bar (≥0.03 lift, p < 0.01)
- ❌ No GUI, dashboards, or interactive visualizations
- ❌ No ML integration endpoints until core package is stable

---

## Phase 3: Service Layer (if warranted)

**Prerequisite:** Phase 2 complete + demonstrated external demand (≥10 independent users or 3+ citations/forks).

**Goal:** Provide programmatic access without becoming a cloud infrastructure project.

### Deliverables (tentative)

1. **Lightweight API**
   - Flask/FastAPI service wrapping `spectral_disruption` package
   - POST endpoint: submit sequences + efficacy, receive scored JSON
   - Rate limiting: 1,000 queries/day per API key
   - Logging: query count, execution time, errors
   - Deployment: single cloud instance (AWS Lambda, GCP Cloud Run, or equivalent)

2. **CLI enhancement**
   - `score_grna --remote` flag to hit API instead of local computation
   - Local-first by default

3. **Minimal UI (optional)**
   - Single-page app: upload FASTA, download scored CSV
   - No authentication required for public tier (<100 sequences)
   - Premium tier (API key) only if demand justifies maintenance overhead

### Gates

- ✅ API responds in <10s for N=100 sequences
- ✅ 99% uptime over 30-day trial
- ✅ Cost per query <$0.01 (compute + storage)

### Anti-goals

- ❌ No freemium/premium pricing until user base >100 active
- ❌ No HIPAA/GDPR compliance engineering unless a paying customer requires it
- ❌ No proprietary sequence encryption unless explicitly contracted
- ❌ No audit logs, admin dashboards, or enterprise features speculatively

---

## Decision Points

### After Phase 1

- **If k* is dataset-invariant (±0.02):** Proceed to Phase 2 with k=0.300 as default.
- **If k* varies significantly (>0.05):** Document auto-tuning protocol, add k-sweep to package, update success gate to "user-tunable k with guidance."
- **If ΔROC-AUC < 0.03 on any dataset:** Investigate failure mode. Do not proceed to Phase 2 until resolved or failure documented as edge case with explicit scope limit.

### After Phase 2

- **If <10 external users in 6 months:** Archive project, publish results as research note, do not build service layer.
- **If ≥10 users or citations:** Evaluate demand for API. If queries exceed local compute feasibility, proceed to Phase 3 minimally.
- **If competing tools (e.g., RuleSet4, improved DeepHF) surpass performance:** Reassess. Either improve algorithm or sunset project.

---

## Success Criteria (Overall)

1. **Validation:** ΔROC-AUC ≥ 0.03, p < 0.01 on ≥3 independent datasets with N>1,000 each
2. **Reproducibility:** Any researcher can clone repo, run validation script, reproduce results within bootstrap CI
3. **Adoption:** ≥10 independent users or ≥3 citations within 12 months
4. **Efficiency:** Scoring 1,000 sequences completes in <60s on laptop hardware

## Failure Modes (and responses)

- **Signal doesn't generalize:** Publish negative result, document where it works and where it fails, archive repo
- **Computational cost prohibitive:** Optimize FFT path, reduce bootstrap samples to 1,000 if CI width acceptable, or scope to smaller N
- **No external interest:** Treat as personal research tool, do not invest in service layer
- **Competing tool dominates:** Acknowledge, compare openly, either improve or move on

---

## Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Core validation | 8 weeks | 8 weeks |
| Decision point 1 | 1 week | 9 weeks |
| Phase 2: Package and document | 4 weeks | 13 weeks |
| Decision point 2 | 2 weeks | 15 weeks |
| Phase 3: Service layer (if warranted) | 6 weeks | 21 weeks |

**Total to production-ready package:** 13 weeks (3.25 months)  
**Total to service layer (if needed):** 21 weeks (5.25 months)

---

## What This Is Not

- A business plan
- A grant proposal
- A pitch deck
- A feature wishlist
- A commitment to build a company

## What This Is

A test plan. Prove the invariant generalizes. Package it cleanly. Let demand dictate infrastructure. Delete everything that doesn't serve validation, reproducibility, or adoption. The geodesic from hypothesis to proof.
