# Task B — Bioactive protein prioritization

The headline task. Rank microbial protein families by how likely they are to be biologically active in IBD, and beat MetaWIBELE on the same data.

## Status

| Phase | Notebook | Status |
|---|---|---|
| Smoke test on synthetic data | `00_smoke_test.ipynb` | ✅ passes |
| Real data download + labels | `01_data.ipynb` | ✅ done (2.7 GB; processed arrays in `processed/`) |
| v0 abundance-only model | `02_abundance_only.ipynb` | ✅ done — **MLP AUPRC 0.142 vs MetaWIBELE 0.076** (see `../../../RESULTS.md`) |
| v1 + ESM-2 sequence | `03_sequence.ipynb` | ⏳ pending CSUC cluster |
| Cross-cohort (Task C) | separate folder | not started |

Results, tables and figures are documented in the repository-level `RESULTS.md`. Figures live in `figures/`, saved predictions/models in `predictions/`.

## v0 model (this folder)

- **Input per family:** abundance vector across all HMP2 samples (length ≈ 1,638)
- **Output:** scalar score in [0, 1] = predicted P(bioactive)
- **Model:** small MLP — 2 hidden layers, BCE loss with positive-class weighting (positives are rare, ~10%)
- **Baselines we must beat:**
  1. Random ranking — pure sanity check
  2. "Ecology score" — mean(abundance in dysbiotic) − mean(abundance in non-dysbiotic). This is the *simplified* MetaWIBELE signal (the full priority score also uses homology + structure + phenotype stats)
  3. ElasticNet on the abundance vector — strong linear baseline
- **Metrics:** AUPRC, Precision@K (K = 10, 50), enrichment vs random

## v1 model (cluster)

Add ESM-2 sequence embeddings as a second input branch, fuse with abundance branch, retrain. ESM-2 35M can run on MPS for a few thousand families; 650M needs the cluster for ~1M families.

## Double ML — when (not) to use

Skip for v0 and v1. Double/Debiased ML is the right tool for *causal* questions like "after controlling for parent species abundance, is this protein's own enrichment in dysbiotic samples predictive of bioactivity?". Task B as framed is *predictive ranking*, not causal estimation — MetaWIBELE is also predictive, so the comparison stays apples-to-apples. Revisit DML only if the supervisor or reviewers ask for a causal claim.

## Compute plan

| Stage | Where | Why |
|---|---|---|
| 00 smoke test | MacBook (MPS) | seconds — pipeline correctness |
| 01 data download | MacBook | one-time, network bound |
| 02 abundance model | MacBook (MPS) | minutes — input is small |
| 03 sequence (ESM-2 650M) | CSUC cluster | hours of GPU time |
| 04 hyperparam sweep + seeds | CSUC cluster | embarrassingly parallel |

See `../../../cluster_access.md` for CSUC connection details.
