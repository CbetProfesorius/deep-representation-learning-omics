# Deep Representation Learning for High-Dimensional Omics Data — Experimental Log

**Author:** Arijus Skaisgirys
**Reference paper reproduced/challenged:** Zhang et al., *Discovery of bioactive microbial gene products in inflammatory bowel disease*, Nature **606**, 754–760 (2022). Method = MetaWIBELE.
**Dataset:** HMP2 / IBDMDB — 1,595 stool metagenomes from 130 individuals (Crohn's disease, ulcerative colitis, non-IBD controls), longitudinal over ~1 year.
**Compute:** MacBook (Apple MPS) for Task A and the abundance models; CSUC HPC cluster (NVIDIA H100) for the ESM-2 protein-language-model embeddings in Task B v1.

> This file is the running record of every experiment, table, figure and finding produced so far. It is written so it can be pasted into the thesis Methods/Results chapters with minimal editing. All numbers are produced by the notebooks in `data/tier_1/`, not hand-copied; re-running the notebooks regenerates them.

---

## 0. Summary of headline findings

1. **Task A (sample-level dysbiosis classification).** On HMP2, classical linear models on community-level functional profiles are strong: ElasticNet on pathway abundances reaches **test AUROC 0.940 / AUPRC 0.866**. A small autoencoder + logistic-regression head **does not beat** the linear baselines on these feature sizes (≤466 features), which is the expected behaviour for autoencoders in the low-sample / moderate-dimension regime and is itself a reportable result.
2. **Task B (bioactive protein-family prioritization — the headline task).** On 1,447,952 protein families with 27,034 metaproteomics-validated positives (1.87 % base rate), a small MLP on the abundance profile reaches **test AUPRC 0.142**, versus the **published MetaWIBELE unsupervised priority's 0.076 on the identical test families** — a **+0.067 absolute / ≈1.9× relative** improvement. Even a plain linear classifier (0.104) beats MetaWIBELE.
3. The MLP beats MetaWIBELE in **every** protein-characterization stratum (characterized, weak-homology, novel), with the largest *relative* lift on weak-homology families (≈2.5×) — the under-annotated proteins this thesis targets.
4. An important honesty caveat is documented (label–feature circularity via abundance) that motivates the sequence-only experiments in Task B v1.
5. **Task B v1 (sequence).** A sequence-only ranker is **abundance-independent**, which neutralizes the circularity caveat. With amino-acid k-mer composition it reaches **test AUPRC 0.182**; with **ESM-2 650M protein-language-model embeddings computed on the CSUC H100 cluster** it reaches **test AUPRC 0.271 / AUROC 0.910** (the 35M model: 0.241), both beating the abundance model (0.142) and MetaWIBELE (0.076). Fusing abundance + ESM-2 650M sequence (leakage-free, validation-fitted) reaches **test AUPRC 0.368 / AUROC 0.927**, **2.6× the abundance model and 4.8× MetaWIBELE**, with **86 % precision in the top 100** (46× enrichment). This is the strongest result of the thesis and establishes that sequence carries substantial signal *beyond* abundance.

---

## 1. Task A — Sample-level phenotype classification

### 1.1 Goal

Predict whether a stool sample is **dysbiotic** vs **non-dysbiotic** (the supervised label used by Zhang 2022) from its microbial community profile. This reproduces the paper's sample-level analysis and serves as the testbed for the thesis question: *do learned (deep) representations beat classical dimensionality reduction on omics matrices?*

### 1.2 Data and features

| Feature set | # features used | Source matrix |
|---|---|---|
| Species (taxonomic) | 572 species | `taxonomic_profiles.tsv.gz` (MetaPhlAn relative abundances, species rows only) |
| Pathways (functional) | 466 community-level pathways | `pathabundance_relab.tsv.gz` (HUMAnN MetaCyc, community rows only, `|`-stratified rows dropped) |

Label: `is_dysbiotic` from `dysbiosis_scores.tsv` (1,594 scored samples). Matrices are stored features × samples and transposed to samples × features for modelling.

### 1.3 Critical evaluation protocol

**Splits are by `Participant ID`, never by sample.** Each patient contributes many longitudinal samples; splitting by sample would leak a patient across train/test and inflate AUROC. Realized split:

| Split | Participants | Samples | Dysbiotic rate |
|---|---|---|---|
| train | 78 | 977 | 0.163 |
| val | 26 | 343 | 0.137 |
| test | 26 | 275 | 0.240 |

(130 participants total, 57 ever-dysbiotic.) The test set's higher dysbiotic rate (0.240) reflects patient-level imbalance and is left as-is to avoid leakage.

### 1.4 Models

- **ElasticNet** — logistic regression with combined L1/L2 penalty on log-transformed abundances. The paper's own baselines are linear models, so this is the like-for-like comparison.
- **PCA(32) + Logistic Regression** — classical linear dimensionality reduction, the standard baseline the thesis must compare deep representations against.
- **Autoencoder(bottleneck=32) + Logistic Regression** — a small PyTorch AE (`in→256→64→32→64→256→in`, ReLU, MSE reconstruction) trained unsupervised; its 32-dim bottleneck is then fed to logistic regression. This is the "deep representation" under test.

### 1.5 Results (Task A)

`data/tier_1/task_a/processed/task_a_results.csv`, visualized in `data/tier_1/task_a/task_a_results.png`.

| Method | Feature set | val AUROC | val AUPRC | val F1 | **test AUROC** | **test AUPRC** | **test F1** |
|---|---|---|---|---|---|---|---|
| ElasticNet | species | 0.777 | 0.397 | 0.505 | 0.857 | 0.694 | 0.662 |
| **ElasticNet** | **pathways** | **0.903** | **0.581** | **0.608** | **0.940** | **0.866** | **0.733** |
| PCA32 + LR | species | 0.849 | 0.605 | 0.642 | 0.612 | 0.507 | 0.490 |
| PCA32 + LR | pathways | 0.882 | 0.710 | 0.617 | 0.885 | 0.738 | 0.654 |
| AE32 + LR | species | 0.699 | 0.428 | 0.462 | 0.495 | 0.407 | 0.349 |
| AE32 + LR | pathways | 0.839 | 0.455 | 0.563 | 0.809 | 0.655 | 0.581 |

### 1.6 Findings (Task A)

- **Functional (pathway) profiles beat taxonomic (species) profiles** across every model — consistent with the IBD-microbiome literature that function is more reproducible than taxonomy.
- **The autoencoder does not beat the linear baselines** at these dimensionalities. AE32+LR on pathways (test AUROC 0.809) trails both ElasticNet (0.940) and PCA32+LR (0.885). This is expected: with only 977 training samples and ≤466 features, an unsupervised reconstruction bottleneck discards label-relevant variance that a supervised linear model keeps. **This is a legitimate negative result** and directly frames the thesis narrative — deep representation learning needs either (a) far higher dimensionality (millions of gene families, Task B) or (b) self-supervision at scale before it pays off over classical linear methods.
- PCA32+LR on species **overfits**: val AUROC 0.849 collapses to test 0.612, a textbook small-sample instability that the thesis can use to argue for more robust representations.

### 1.7 Task A figures

- `data/tier_1/task_a/explore_overview.png` — PCA/UMAP projections of samples coloured by dysbiosis.
- `data/tier_1/task_a/task_a_results.png` — bar chart of the table above.

---

## 2. Task B — Bioactive protein-family prioritization (headline task)

### 2.1 Goal

Rank ~1.4 M microbial protein families by predicted bioactivity in IBD, and **beat the published MetaWIBELE priority scores on the same families and the same positive labels.**

### 2.2 Pipeline / notebooks

| Stage | Notebook | Status |
|---|---|---|
| Smoke test (synthetic data, pipeline correctness) | `00_smoke_test.ipynb` | ✅ passes |
| Real data download + label construction | `01_data.ipynb` | ✅ done (2.7 GB) |
| **v0 abundance-only model** | `02_abundance_only.ipynb` | ✅ done (this section) |
| **v1 sequence (k-mer + ESM-2) and fusion** | `03_sequence.ipynb` | ✅ done — ESM-2 35M + 650M on CSUC H100 |

### 2.3 Data construction (`01_data.ipynb`)

Downloaded from the Huttenhower MetaWIBELE data portal (`http://199.94.60.28/MetaWIBELE_data/HMP2/`) and the Nature supplementary bundle:

| File | Size | Role |
|---|---|---|
| `HMP2_proteinfamilies_nrm.tsv.gz` | 1.6 GB | abundance matrix (families × samples) |
| `HMP2_proteinfamilies.centroid.faa.gz` | 283 MB | representative protein sequences (for ESM-2 in v1) |
| `HMP2_proteinfamilies.clstr.gz` | 30 MB | cluster → representative-sequence id map |
| `HMP2_proteinfamilies_annotation.tsv.gz` | 707 MB | homology / taxonomy / characterization |
| `HMP2_unsupervised_prioritization.rank.table.tsv.gz` | 79 MB | MetaWIBELE baseline 1 (ecology only) |
| `HMP2_supervised_prioritization.rank.table.tsv.gz` | 140 MB | MetaWIBELE baseline 2 (ecology + phenotype) |
| `41586_2022_4648_MOESM4_ESM.xlsx` | 9.4 MB | Supplementary Tables S2–S33 (positive labels) |

**Positive labels** = metaproteomics (MPX) -validated protein families: union of
- **Supplementary Table S7** (`familyID` column — every family detected by mass spectrometry), and
- **Supplementary Table S8** (`core_enrichment` field of the GSEA result, slash-separated cluster IDs).

This yields **27,548** positive family IDs; after restricting to families present in the abundance matrix, **27,034** positives remain.

**Key schema gotchas solved (documented for reproducibility):**
- The priority tables are **long-format** (one row per `familyID` × `evidence` type). Only `evidence == 'priority_score'` rows are the published meta-rank. For the supervised table each family appears once per phenotype contrast (CD-dysbiosis, UC-dysbiosis); the **max rank across contrasts** is used.
- FASTA headers are representative-sequence names (e.g. `ESM5MEB9_P_71053`), **not** cluster IDs (`Cluster_1`). The `.clstr` file maps the two; without it, sequence joins silently fail (every sequence empty).

**Final processed artefacts** (`data/tier_1/task_b/processed/`):

| File | Size | Contents |
|---|---|---|
| `abundance.npy` | 8.6 GB | 1,447,952 families × 1,595 samples, float32 |
| `sequences.parquet` | 398 MB | all 1.45 M families; median length 223 aa, max 12,684 aa |
| `labels.parquet` | 16 MB | family_id, is_bioactive, metawibele_unsup_rank, metawibele_sup_rank |
| `annotations.parquet` | 17 MB | strong/weak UniRef90 homology hit (NaN+NaN ⇒ novel) |
| `family_ids.parquet`, `sample_ids.json` | — | row/column order for `abundance.npy` |

### 2.4 v0 model and protocol (`02_abundance_only.ipynb`)

- **Input per family:** its abundance profile across the 1,595 samples (1,595-dim vector), `log1p`-transformed and per-feature standardized (statistics fit on train rows only).
- **Split:** stratified on `is_bioactive`, by family — 70 / 15 / 15 → train 1,013,566 / val 217,193 / test 217,193, base rate 0.0187 in all three.
- **Models:**
  - Linear: `Linear(1595 → 1)` (logistic regression in torch).
  - MLP: `Linear(1595→256) → ReLU → Dropout(0.3) → Linear(256→64) → ReLU → Linear(64→1)`.
  - Loss: `BCEWithLogitsLoss` with `pos_weight = 52.56` (neg/pos ratio) to handle the 1.87 % imbalance; Adam; early-stop on val AUPRC; 6 epochs (linear) / 8 epochs (MLP).
- **Baselines on the identical test families:** Random; an "Ecology score" = harmonic mean of mean-abundance and prevalence (reproduces MetaWIBELE-unsupervised logic on our split); and the two **published** MetaWIBELE priority scores.

### 2.5 Results (Task B v0)

`data/tier_1/task_b/predictions/02_results_table.csv`.

| Method | AUPRC | AUROC | P@100 | enr@100 | P@1000 | enr@1000 |
|---|---|---|---|---|---|---|
| Random | 0.018 | 0.497 | 0.00 | 0.0× | 0.012 | 0.6× |
| Ecology score (harm. mean abu, prev) | 0.075 | 0.746 | 0.35 | 18.7× | 0.210 | 11.2× |
| MetaWIBELE supervised (published) | 0.064 | 0.703 | 0.32 | 17.1× | 0.198 | 10.6× |
| **MetaWIBELE unsupervised (published)** | **0.076** | 0.730 | 0.35 | 18.7× | 0.212 | 11.4× |
| Linear (LogReg on abundance) | 0.104 | 0.797 | 0.34 | 18.2× | 0.264 | 14.1× |
| **MLP on abundance (ours)** | **0.142** | **0.832** | **0.41** | **22.0×** | **0.339** | **18.2×** |

- **Best AUPRC: MLP = 0.142.** MetaWIBELE-unsup on the same split = 0.076.
- Linear vs MetaWIBELE: **+0.029**; MLP vs MetaWIBELE: **+0.067** (≈1.9×).

### 2.6 Stratified analysis by characterization category

`data/tier_1/task_b/predictions/02_results_stratified.csv`. Families bucketed using `annotations.parquet`: **characterized** (strong UniRef90 homology), **weak** (only weak homology), **novel** (no homology — the families MetaWIBELE was built to surface).

| Bucket | n | positives | base rate | MetaWIBELE unsup AUPRC | MetaWIBELE sup AUPRC | **MLP AUPRC** |
|---|---|---|---|---|---|---|
| characterized | 125,750 | 3,302 | 0.0263 | 0.096 | 0.079 | **0.167** |
| weak homology | 67,135 | 703 | 0.0105 | 0.024 | 0.023 | **0.059** |
| novel (no homology) | 24,308 | 50 | 0.0021 | 0.003 | 0.003 | **0.005** |

- The MLP **wins in every bucket.**
- Largest **relative** lift is on **weak-homology** families (0.024 → 0.059, ≈2.5×) — exactly the under-annotated proteins of interest.
- The **novel** bucket has only 50 positives (extremely sparse); no model does well and no firm conclusion can be drawn. **Sequence features (ESM-2, v1) are the intended lever here.**

### 2.7 Figures (Task B)

| File | Figure |
|---|---|
| `data/tier_1/task_b/figures/fig1_pr_curves.png` | Precision–Recall curves, all methods |
| `data/tier_1/task_b/figures/fig2_roc_curves.png` | ROC curves, all methods |
| `data/tier_1/task_b/figures/fig3_auprc_bars.png` | AUPRC bar chart (headline figure) |
| `data/tier_1/task_b/figures/fig4_stratified.png` | AUPRC by characterization category |
| `data/tier_1/task_b/figures/fig5_enrichment_at_k.png` | Enrichment@K (log-x), all methods |

### 2.8 Findings & caveats (Task B v0)

1. **A learned non-linear function of the full abundance profile extracts substantially more signal than MetaWIBELE's two aggregate ecology statistics.** MetaWIBELE's unsupervised score is essentially `harmonic_mean(prevalence, mean_abundance)`; the MLP, which sees the per-sample profile (who carries the family, in what amount, implicitly in which disease state), nearly doubles AUPRC. This is the core positive result of the chapter.
2. **Interestingly, the unsupervised MetaWIBELE score beats the supervised one** against MPX positives (0.076 vs 0.064). MPX detection ("actually translated and abundant enough to be seen by mass-spec") is ecology-driven, not phenotype-contrast-driven — worth a sentence of discussion.
3. **Honesty caveat — label/feature circularity.** Positives are MPX-detected families, and detectability by mass-spec correlates with abundance; the model's input *is* abundance. So part of the lift is "abundant families are easier to detect." This confound applies **equally to MetaWIBELE** (its score is also abundance-derived), so the head-to-head comparison stays fair — but the absolute AUPRC should not be read as "predicting bioactivity from first principles." The clean test of whether sequence/function (independent of abundance) carries signal is the ESM-2 model in v1.
4. **Minor data note:** the abundance matrix is 1,595 samples (not the 1,638 quoted in some HMP2 summaries) after MetaWIBELE's own sample QC; use 1,595 in the writeup.

---

## 2bis. Task B v1 — sequence model and abundance+sequence fusion

### 2bis.1 Question

Does protein **sequence** carry bioactivity signal **beyond** the abundance profile? This is the clean test the abundance model (§2.8 caveat 3) could not provide: a sequence-derived score is independent of the abundance/label circularity.

### 2bis.2 Setup (`03_sequence.ipynb`)

- **Identical family split as v0** (same `random_state=0`); verified programmatically (`test split matches 02 predictions: True`), so all comparisons are on the same 217,193 test families at the true 1.87 % base rate.
- **Embedding backend is swappable.** Two are reported: **(a) amino-acid k-mer composition** (k=2 → 400-dim normalized frequency), a classical baseline that runs on the laptop; and **(b) ESM-2** (`facebook/esm2_t12_35M_UR50D`, 480-dim) protein-language-model embeddings, mean-pooled over the last hidden state, **computed for all 1.45 M families on the CSUC H100 cluster** (≈1.3 h, see §2bis.5) and loaded via `EMBED_BACKEND='precomputed'`.
- **Three models:** (1) sequence-only MLP (`d→256→64→1`); (2) abundance-only v0, reloaded from `predictions/02_mlp.pt` and re-scored with the same standardization; (3) **fusion** = logistic regression on `[abundance_score, sequence_score]` fit on the **validation** split and evaluated on test (no leakage).

### 2bis.3 Results (Task B v1)

`data/tier_1/task_b/predictions/03_results_kmer.csv` and `03_results_precomputed.csv` (ESM-2 35M).

: Table 4 — Task B v1 (test set: 217,193 families, base rate 1.87 %).

| Method | AUPRC | AUROC | P@100 | enr@100 | P@1000 | enr@1000 |
|---|---|---|---|---|---|---|
| MetaWIBELE supervised | 0.064 | 0.703 | 0.32 | 17.1× | 0.198 | 10.6× |
| MetaWIBELE unsupervised | 0.076 | 0.730 | 0.35 | 18.7× | 0.212 | 11.4× |
| Abundance v0 (MLP) | 0.142 | 0.832 | 0.41 | 22.0× | 0.339 | 18.2× |
| Sequence-only (k-mer) | 0.182 | 0.875 | 0.51 | 27.3× | 0.351 | 18.8× |
| Sequence-only (ESM-2 35M) | 0.241 | 0.902 | 0.61 | 32.7× | 0.452 | 24.2× |
| **Sequence-only (ESM-2 650M)** | **0.271** | **0.910** | 0.73 | 39.1× | 0.481 | 25.8× |
| Fusion (abundance + k-mer) | 0.307 | 0.911 | 0.75 | 40.2× | 0.583 | 31.2× |
| Fusion (abundance + ESM-2 35M) | 0.359 | 0.925 | 0.91 | 48.7× | 0.641 | 34.3× |
| **Fusion (abundance + ESM-2 650M)** | **0.368** | **0.927** | **0.86** | **46.1×** | **0.631** | **33.8×** |

![Task B v1: AUPRC progression from MetaWIBELE through abundance, sequence (k-mer, ESM-2) and fusion.](data/tier_1/task_b/figures/fig6_v1_auprc.png)

![Task B v1: Precision-Recall curves for the ESM-2 sequence-only and fusion models vs the abundance baseline.](data/tier_1/task_b/figures/fig7_v1_pr.png)

### 2bis.4 Findings (Task B v1)

1. **Sequence alone beats abundance alone and crushes MetaWIBELE.** Even k-mer composition (0.182) beats the abundance MLP (0.142); **ESM-2 650M embeddings lift the sequence-only model to AUPRC 0.271 / AUROC 0.910** (35M: 0.241) — 3.6× MetaWIBELE's 0.076. This is the central evidence that sequence carries independent bioactivity signal, and model scale adds a clear gain over both k-mer and 35M.
2. **The sequence score sidesteps the abundance/label circularity** (§2.8 caveat 3): it never sees abundance, so its lift cannot be attributed to "abundant proteins are easier to detect by mass-spec."
3. **Fusion is the best model by a wide margin.** Abundance + ESM-2 650M reaches **AUPRC 0.368 / AUROC 0.927**, 2.6× the abundance model and **4.8× MetaWIBELE**, with **86 % precision in the top 100** (46× enrichment) and 63 % at top 1000. The two modalities are complementary: abundance captures ecological prevalence, sequence captures compositional/functional identity — exactly the regime where the top-K precision matters for wet-lab candidate selection.
4. **ESM-2 > k-mer at every operating point, and 650M > 35M**, confirming that evolutionary/structural context and model scale beyond local composition help.

### 2bis.5 Cluster path (ESM-2, done)

ESM-2 embeddings were computed on the **CSUC Pirineus3** cluster (Slurm, NVIDIA **H100**). Because the shared `work_env` is read-only and lacks PyTorch, a personal conda env (`$DATA/.conda/envs/taskb`, torch 2.6+cu124 / transformers 5.9 / scikit-learn 1.9) was built through the site HTTP proxy; model weights were prefetched into `$DATA/hf_cache` so compute nodes run fully offline (`HF_HUB_OFFLINE=1`). `esm_embed.py` (resumable, fp16, memmap) mean-pools the last hidden state for all 1,447,952 families: the **35M** model in **1 h 18 m (≈317 seq/s)** → `seq_emb_35M.npy` (480-dim, 2.8 GB), the **650M** model in **4 h 32 m (≈89 seq/s)** → `seq_emb_650M.npy` (1280-dim, 7.4 GB). Because the 7.4 GB matrix exceeds the laptop's free disk, the 650M sequence + fusion eval runs **on the cluster** (`eval650.py` via `run_eval650.sh`): abundance val/test scores are precomputed locally (`compute_abund_scores.py`) and shipped as a 5 MB file, and only the small result CSV/parquet are copied back. The 35M result was produced locally via `03_sequence.ipynb` with `EMBED_BACKEND='precomputed'`. Note the `gpu` partition uses per-GPU memory defaults, so `--mem`/`--cpus-per-task` must be omitted and `--gres=gpu:h100:1 --account=uvicommsc_normal --qos=normal` used.

### 2bis.6 Hardening & exploration (done)

**Multi-seed CIs (`06_seeds.py` → `06_seed_summary.csv`, `fig12_seed_ci.png`).** Reran abundance MLP + ESM-2 35M sequence MLP + fusion from scratch on 5 stratified splits (seeds 0–4), refitting standardization/weights/fusion each time. Mean test AUPRC ± 95% CI: MetaWIBELE sup 0.067 [0.065,0.069], MetaWIBELE unsup 0.078 [0.075,0.081], Abundance MLP 0.140 [0.137,0.144], Sequence 35M 0.227 [0.223,0.231], **Fusion 0.348 [0.341,0.356]**. Every model's CI is fully separated from MetaWIBELE's; fusion ranges 0.339–0.360 across seeds. The "beats MetaWIBELE" claim is not seed noise. (35M used so the small matrix can be reused across splits on the laptop; 650M single-run is higher.)

**Stratified sequence analysis (`05_stratified_seq.py` → `05_stratified_seq.csv`, `fig10_stratified_seq.png`).** AUPRC by characterization bucket for the ESM-2 650M models:

| bucket | n | pos | base | MW unsup | Abund MLP | Seq 650M | Fusion |
|---|---|---|---|---|---|---|---|
| characterized | 125,750 | 3,302 | 0.026 | 0.096 | 0.167 | 0.301 | **0.403** |
| weak | 67,135 | 703 | 0.011 | 0.024 | 0.059 | 0.184 | **0.212** |
| novel | 24,308 | 50 | 0.002 | 0.003 | 0.005 | 0.074 | **0.091** |

On **novel** (no-homology) families the abundance MLP basically failed (0.005, near the 0.002 base rate); the sequence model reaches 0.074 (~14× abundance, ~22× MetaWIBELE) and fusion 0.091 (~43× base rate). The relative gains are largest exactly where annotation/abundance methods are weakest — the case for a sequence model.

**Complementarity (`07_complementarity.py` → `07_complementarity.csv`, `fig11_complementarity.png`).** Spearman(abundance, sequence) = 0.232 on the test set; top-1000 Jaccard 0.022. Of the true positives in each model's top 1000, abundance recovers 339, sequence 481, only 41 shared (union 779). The two signals are nearly independent and recover largely different positives — why the fusion gain is so large.

---

## 2ter. Task C — cross-cohort generalization (done)

**Goal.** The thesis hypothesis is about generalization, so train IBD (CD/UC) vs non-IBD-control on HMP2 and test, no refitting, on an independent cohort: Franzosa et al. 2019 PRISM+validation (220 samples: 88 CD, 76 UC, 56 control), from the paper's Supplementary Dataset 4 (`raw/moesm6.xlsx`). Both cohorts use MetaPhlAn2 → **197 shared species**. Relative abundances re-closed on shared species, log10, standardized on the training cohort. Base rates almost equal (HMP2 73.3% IBD, Franzosa 74.5%). Models: ElasticNet, PCA32+LR, Autoencoder32+LR. (`taskc_crosscohort.py`)

**The leakage trap (`taskc_leakage.csv`).** ElasticNet on HMP2 species, IBD vs control: naive sample-level 5-fold CV = **AUROC 0.948**, but participant-grouped 5-fold CV = **0.460** (chance). HMP2 is longitudinal; naive CV memorizes individuals. Concrete justification for participant/cohort-level splits everywhere (§2.2).

**Cross-cohort transfer (`taskc_results.csv`, `fig13`, `fig14`).**

| Model | within-cohort CV (clean, Franzosa) | HMP2→Franzosa AUROC | HMP2→Franzosa AUPRC | Franzosa→HMP2 AUROC | mean transfer |
|---|---|---|---|---|---|
| ElasticNet | 0.844 | 0.671 | 0.855 | 0.728 | 0.699 |
| PCA32+LR | 0.890 | 0.767 | 0.915 | 0.702 | 0.735 |
| **Autoencoder32+LR** | 0.865 | **0.790** | **0.927** | 0.697 | **0.744** |

**Findings.**
1. All three transfer well above chance (AUROC 0.67–0.79, AUPRC 0.83–0.93) — a model trained on one cohort does carry to another.
2. **Data-rich direction (train big HMP2 → test Franzosa): the deep representation wins outright** — autoencoder 0.790 / PCA 0.767 vs ElasticNet 0.671 (+0.12 AUROC, +0.07 AUPRC). First clean win for DL in the thesis, on the generalization metric the hypothesis targets.
3. Reverse (train small 220-sample Franzosa → test HMP2) is even: ElasticNet 0.728 ≥ PCA 0.702 ≥ AE 0.697 — a 32-dim embedding needs enough samples to estimate. Mean over both directions still favors AE/PCA.
4. Cross-cohort costs ~0.07–0.17 AUROC vs the clean within-cohort reference (0.84–0.89). Honest conditional: DL representation generalizes better **when there is enough data to learn it**; otherwise the linear model is the safe default.

---

## 2quater. Robustness + deep encoders (Task A, done)

**Setup (`task_a/robust_vae.py`).** HMP2 community pathways (466), dysbiosis label, same participant split. Trained to convergence (300 ep) with balanced LR head. Models: ElasticNet, PCA32+LR, AE32+LR, **Denoising AE32+LR**, **VAE32+LR**. Corrupt only test inputs: additive Gaussian noise (σ grid) and random feature missingness (impute to train mean), AUROC averaged over 20 draws.

**Clean vs robust (test AUROC):**

| Model | clean | noise σ=3 | 70% missing |
|---|---|---|---|
| ElasticNet | 0.963 | 0.765 | 0.835 |
| PCA32+LR | 0.951 | 0.871 | 0.919 |
| AE32+LR | 0.964 | 0.866 | 0.911 |
| Denoising AE32+LR | 0.951 | 0.892 | 0.921 |
| **VAE32+LR** | **0.965** | **0.892** | **0.933** |

**Findings.** (1) With enough training the AE matches ElasticNet on clean data (~0.95), so the §3.2 gap was partly a training-budget artefact. (2) **Objective 3 win:** under corruption the linear model is far more brittle (0.765 at σ=3) than every learned/DR representation; **VAE most robust** (0.892 / 0.933). (3) Objective 1: AE + denoising-AE + VAE implemented. Figures `fig15_robustness.png`, `fig16_clean_models.png`.

## 2quinquies. EC high-dim cross-cohort (Task C companion, done)

**Setup (`task_c/ec_crosscohort.py`).** HMP2 community EC (HUMAnN `ecs_relab`) vs Franzosa enzymes (Supplementary Dataset 6, `moesm8.xlsx`); **2,052 shared EC numbers** (~10× species). Same transfer protocol; EN/PCA/AE/VAE.

| Model | within CV | HMP2→Fr AUROC | HMP2→Fr AUPRC | Fr→HMP2 |
|---|---|---|---|---|
| ElasticNet | 0.905 | 0.704 | 0.873 | 0.690 |
| PCA32+LR | 0.901 | 0.704 | 0.883 | 0.667 |
| **AE32+LR** | 0.908 | **0.738** | **0.898** | 0.653 |
| VAE32+LR | 0.867 | 0.687 | 0.865 | 0.629 |

**Findings.** AE again transfers best in the data-rich direction (0.738 vs 0.704), confirming the species result on a 10× larger feature space, though the margin is smaller (+0.034 vs +0.12 species) — EC profiles more redundant. VAE doesn't help on this redundant input. Figure `fig17_ec_crosscohort.png`.

---

## 2sexies. Biological interpretation (done)

**Task B — what the top-ranked families are (`task_b/extract_annotations.py`, `interpret_taskb.py`).** Annotated test families with MetaWIBELE functional layer (Pfam, PSORTb, taxonomy, MaAsLin2). Top-1000 by fusion score vs background, Fisher + BH-FDR:
- **Pfam:** top 5 are all SusC/SusD TonB-dependent transporters (PF07715/PF00593 TonB receptor, PF14322/PF07980 SusD; OR 13–16, FDR<1e-26) → Bacteroidetes glycan-foraging machinery. Also rubrerythrin (oxidative stress, OR 40), and elongation factors / central metabolism (abundance/detectability bias, honest caveat).
- **Genus:** Prevotella OR=14 (FDR~1e-132), Bacteroides 3.2, Parabacteroides 3.2.
- **Localization:** enriched extracellular (OR 1.7) + periplasmic (OR 4.1), depleted inner-membrane → secreted/surface, host-interacting.
- **Validation:** 82% of top families are MaAsLin2 differentially-abundant in dysbiosis vs 38% background (OR 7.5, p<1e-179). Ranking concentrates real IBD-associated families. Fig `fig19_interpret_taskb.png`.

**Task C — taxa driving IBD (`task_c/feature_importance.py`).** Per-species standardized IBD−control difference (Cohen's d) in each cohort; cross-cohort **r=0.68**, known-signature direction agreement **0.85**. IBD-up: *Ruminococcus gnavus*, *Clostridium* XIVa (clostridioforme/bolteae/symbiosum). IBD-down (depleted): *Roseburia hominis*, *Subdoligranulum*, *Alistipes*, *Dorea* (SCFA producers). → transfer works because the signal is reproducible biology, not batch effect. Fig `fig18_feature_importance.png`.

Note: ElasticNet coefficients were unstable across cohorts (r=0.11) due to elastic-net selecting among correlated species; switched to univariate differential abundance for stable, interpretable effects.

---

## 2septies. Error measurements + PCA ordination (done)

Added in response to "more literature sources, interpretations like PCA, errors / measurements of them".

**Task B bootstrap CIs (`task_b/bootstrap_ci.py`, fig20).** B=1000 resamples of the fixed 217,193-family 650M test set, model held constant → pure evaluation noise. AUPRC [95% CI]: MetaWIBELE unsup 0.075 [0.070, 0.082], MW sup 0.064 [0.059, 0.070], abundance MLP 0.142 [0.133, 0.152], sequence 650M 0.271 [0.257, 0.286], fusion 0.368 [0.353, 0.385]. **All intervals non-overlapping** → every rung of the ladder is statistically clear. Complements the §2bis.6 multi-seed CIs (which vary the train split); both agree.

**Task C transfer CIs (`task_c/taskc_crosscohort.py` `boot_ci`, error bars on fig13).** B=2000 resamples of the 220-sample Franzosa test set. HMP2→Franzosa AUROC: ElasticNet 0.671 [0.585, 0.746], PCA 0.767 [0.700, 0.828], AE 0.790 [0.728, 0.849]. AE/PCA CIs overlap (gap not significant) but both clear ElasticNet's point estimate → honest claim is "AE≈PCA > ElasticNet", not "AE > PCA".

**PCA ordination of pooled cohorts (`task_c/ordination.py`, fig21).** Standardized log-abundance, 197 shared species, 10-PC PCA on pooled HMP2+Franzosa. PC1 8.9%, PC2 5.8% variance. 5-fold logistic on 10 PCs: **cohort/batch AUROC 0.99**, **diagnosis AUROC 0.66**. → batch effect dominates the variance; biology is a faint overlapping axis. Makes the 0.79 transfer more impressive (model keys on the faint shared axis, not the loud batch one) and motivates training-cohort standardization. PCA does double duty: classifier baseline + confound-vs-signal diagnostic.

**Literature.** Expanded References 9 → 33 with a new §1.4 Related work (MetAML/Pasolli, DeepMicro/Oh&Zhang, curatedMetagenomicData, AE/DAE/VAE, PCA Jolliffe, t-SNE/UMAP, ESM-1b/ESM-2/ProtTrans/AlphaFold, IBD biology Gevers/Sokol/Hall, MaAsLin2/Pfam/InterProScan/PSORTb/UniRef, Saito&Rehmsmeier PR-curve, Efron bootstrap). Inline author-year citations threaded through methods, related work, ordination, and bootstrap sections.

---

## 3. Reproducibility

| What | Where |
|---|---|
| Task A end-to-end | `data/tier_1/task_a/explore.ipynb` |
| Task B smoke test | `data/tier_1/task_b/00_smoke_test.ipynb` |
| Task B data build | `data/tier_1/task_b/01_data.ipynb` (set `SMOKE=True` for a 230 MB dry run) |
| Task B v0 model | `data/tier_1/task_b/02_abundance_only.ipynb` |
| Task B v1 sequence + fusion | `data/tier_1/task_b/03_sequence.ipynb` |
| Cluster ESM-2 embedding | `data/tier_1/task_b/esm_embed.py`, `run_esm_cluster.sh` (35M), `run_esm_650.sh` (650M) |
| v1 figures (ESM-2) | `data/tier_1/task_b/make_v1_figs.py` |
| Multi-seed CIs / stratified seq / complementarity | `06_seeds.py`+`make_ci_fig.py`, `05_stratified_seq.py`, `07_complementarity.py` |
| Task C cross-cohort (HMP2↔Franzosa) | `data/tier_1/task_c/taskc_crosscohort.py` (+ `raw/moesm6.xlsx`) |
| Saved predictions / models / tables | `data/tier_1/task_b/predictions/` |
| Figures | `data/tier_1/task_b/figures/`, `data/tier_1/task_a/*.png` |
| Environment | conda env `masters`, Python 3.12.13, PyTorch (MPS), scikit-learn, pandas, pyarrow, openpyxl |

All notebooks are deterministic (`torch.manual_seed(0)`, `np.random.seed(0)`, fixed `random_state`).

---

## 4. Next steps

1. **Task B v1 ESM-2 (CSUC cluster):** done — both 35M and 650M passes complete (Table 4); 650M is the headline (sequence-only 0.271, fusion 0.368).
2. **Hardening Task B:** done — 5-seed 95% CIs (§2bis.6) are fully separated from MetaWIBELE; stratified-sequence and complementarity analyses added. Longer training / larger sweep optional.
3. **Task C (cross-cohort):** done (§2ter) — HMP2↔Franzosa transfer; AE/PCA beat ElasticNet in the data-rich direction (0.790/0.767 vs 0.671). Optional extension: high-dim EC enzymes (HMP2 `ecs_relab` vs Franzosa `moesm8.xlsx`).
4. **Task A extension:** scale to the high-dimensional EC-number / gene-family matrices where autoencoders are expected to finally pay off, and add a supervised/contrastive encoder (not just unsupervised reconstruction).
