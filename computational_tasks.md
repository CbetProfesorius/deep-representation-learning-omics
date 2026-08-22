# Computational Tasks to Reproduce / Beat with Deep Learning

Reference: Zhang et al., *Discovery of bioactive microbial gene products in inflammatory bowel disease*, Nature 606 (2022). Method = MetaWIBELE.

This is the list of every **computational** thing the paper does. Wet-lab experiments are excluded — they produce ground-truth labels for evaluation, not predictions to reproduce.

---

## Input data the paper uses

- **Metagenomes (MGX):** 1,595 stool DNA samples from 130 people (65 Crohn's, 38 UC, 27 controls), longitudinal, ~1 year.
- **Metatranscriptomes (MTX):** 800 paired RNA samples.
- **Metaproteomes (MPX):** 201 samples, used as validation only.
- **Reference databases:** UniRef90, Pfam, DOMINE, PDB, DEG (essential genes).
- All available at [ibdmdb.org/results](https://ibdmdb.org/results) (processed) and NCBI BioProject [PRJNA398089](https://www.ncbi.nlm.nih.gov/bioproject/398089) (raw).

---

## Tier 1 — Core thesis tasks (must do)

### Task A. Sample-level phenotype classification

| | |
|---|---|
| **Input** | Sample × feature matrix (taxonomic profile, pathway abundance, or gene-family abundance) |
| **Output label** | Dysbiotic vs. non-dysbiotic, or CD vs. UC vs. control |
| **Paper baseline** | Linear models on log-transformed abundances (linear mixed-effects model, FDR-corrected) |
| **My DL methods** | Autoencoder, VAE, contrastive encoder, masked-feature autoencoder → embeddings → classifier head |
| **Classical baselines** | PCA + logistic regression; elastic net; signature scores from literature |
| **Metric** | AUROC, AUPRC, F1 on patient-level held-out splits |
| **Evaluation rule** | Never split a single patient across train/test |

### Task B. Bioactive protein prioritization (the headline task)

| | |
|---|---|
| **Input per protein family** | Sequence + abundance vector across samples + taxonomy + Pfam annotations |
| **Output** | Per-family score = "probability bioactive in IBD" |
| **Paper baseline** | MetaWIBELE priority score (combination of homology + secondary structure + ecology + phenotype stats) |
| **My DL methods** | End-to-end model: ESM-2 sequence embedding + abundance encoder + metadata → ranking head |
| **Positive labels** | Proteomics-validated proteins (Supplementary Tables 7-8) and literature-known IBD-associated proteins |
| **Metric** | Precision@N, AUPRC, enrichment in top-N vs. MetaWIBELE on the same validation set |
| **Sub-versions in the paper** | Unsupervised score (ecology only) + Supervised score (ecology + phenotype) |

### Task C. Cross-cohort generalization

| | |
|---|---|
| **Train on** | HMP2 (IBDMDB) |
| **Test on** | A second IBD cohort — PRISM (Franzosa et al. 2019) is the obvious candidate |
| **Output** | Same as Task A or Task B, but evaluated on a held-out cohort |
| **Metric** | AUROC drop from within-cohort to cross-cohort. The thesis claim is that DL representations transfer better than PCA + elastic net |

---

## Tier 2 — Strong supporting tasks (pick 1–2)

### Task D. Taxonomic annotation by co-abundance ("guilt-by-association")

| | |
|---|---|
| **Input** | Co-abundance graph of protein families across samples |
| **Output** | Predicted taxonomy (species / genus / family) for proteins with no homology hit |
| **Paper baseline** | MSP (Metagenomic Species Pangenome) binning + majority vote |
| **My DL methods** | Graph neural network on the co-abundance graph; or contrastive learning across samples; or VAE clustering |
| **Metric** | Accuracy on the 20% holdout set defined in Extended Data Fig. 4b |

### Task E. Homology characterization (SC / SU / RH / NH classification)

| | |
|---|---|
| **Input** | Protein sequence |
| **Output** | One of 4 labels: Strong-Characterized, Strong-Uncharacterized, Remote-Homology, No-Homology |
| **Paper baseline** | BLAST/DIAMOND vs. UniRef90 |
| **My DL methods** | ESM-2 embedding + k-NN or linear classifier |
| **Metric** | Classification accuracy, F1 per class |

### Task F. Differential abundance per protein

| | |
|---|---|
| **Input** | Protein abundance vector across samples + sample metadata |
| **Output** | Per-protein significance + effect size for dysbiotic vs. non-dysbiotic |
| **Paper baseline** | Linear mixed-effects model with FDR correction (Benjamini-Hochberg) |
| **My DL methods** | Per-feature attribution from a classifier (integrated gradients / SHAP), or learned ranking head |
| **Metric** | Overlap with paper's 348,973 differentially abundant families; FDR control at matched levels |

---

## Tier 3 — Optional extensions

### Task G. Differential expression (metatranscriptomics)

| | |
|---|---|
| **Input** | Paired DNA + RNA abundances per protein family per sample |
| **Output** | Per-family expression change between dysbiotic and non-dysbiotic |
| **Paper baseline** | Metagenome-normalized linear mixed-effects model |
| **My DL methods** | Joint multi-modal encoder over (DNA, RNA) → expression head |

### Task H. Structure prediction for top-prioritized novel proteins

| | |
|---|---|
| **Input** | Protein sequence (e.g., `cluster_37544`, the VWA case in Fig. 4) |
| **Output** | 3D structure |
| **Paper baseline** | Phyre2 + mTM-align (homology-based) |
| **My DL methods** | ESMFold or AlphaFold2 (just running them — no novel research) |
| **Metric** | TM-score / lDDT against reference structures |
| **Effort** | ~1 day; nice as a final visual chapter |

### Task I. Generalization to a different ecosystem (Red Sea marine data)

| | |
|---|---|
| **Input** | Marine metagenomes from Red Sea (BioProject PRJNA289734) |
| **Output** | Same as Task A/B but for epipelagic vs. mesopelagic |
| **Paper baseline** | Same MetaWIBELE pipeline rerun (Extended Data Fig. 10) |
| **My DL methods** | Same model trained on HMP2, fine-tuned or zero-shot on marine |
| **Note** | Stretch goal — only if cross-cohort IBD (Task C) is already done |

---

## Out of scope (computational, but not worth doing for a thesis)

- Pfam domain annotation — solved by InterProScan / DeepFam, no research value to redo.
- Signal peptide / transmembrane prediction — SignalP-6.0 and DeepTMHMM are already SOTA.
- Protein-family clustering — engineering problem, not research.
- BGC detection — already done by antiSMASH / DeepBGC.
- Foldseek structural search — running an existing tool.

---

## Tasks I cannot do (wet-lab — for context only)

These are the paper's experimental validations. I cannot replace them, only **predict them**.

- *E. coli* co-culture with HCT-15 colon cells → IL-8 / GM-CSF expression (Fig. 3e-f).
- `vwf` knockout in *B. fragilis* → biofilm formation in mucin (Fig. 4c).
- Plasmid complementation rescue.
- RT-qPCR confirmation of pilin expression.
- Mass-spectrometry metaproteomics of stool samples.

For my thesis, these results are **positive labels** that my model should recover.

---

## Execution order

1. Task A on the easy feature set (pathways, ~500 features).
2. Task A on the hard feature set (gene families, ~millions of features).
3. Task B with sequence + abundance + metadata input.
4. Task C cross-cohort evaluation of A and B.
5. One of Tasks D-H as a supporting chapter.

## Open questions to confirm with supervisor

1. Do A and B both count as headline contributions, or is one of them enough?
2. Which validation set defines "bioactive" for Task B — only proteomics-validated proteins, or also literature-known IBD genes?
3. Which second cohort for Task C — PRISM, RISK, NLIBD, or something in-house?
4. Is Task H (ESMFold) wanted as a sidebar or a full chapter?
5. Compute resources — laptop only, lab GPU server, or HPC cluster?
