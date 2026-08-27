# Deep Representation Learning for High-Dimensional Omics Data

Master's thesis — MSc in Omics Data Analysis, University of Vic – Central University of Catalonia (UVic-UCC).

**Author:** Arijus Skaisgirys
**Supervisor:** Prof. Dr. Andrius Stasiukynas (Kazimieras Simonavičius University)
**Co-supervisor:** Francesco Strati, PhD (Lithuanian University of Health Sciences)
**Academic tutor:** Jordi Villà i Freixa (Department of Biosciences, UVic-UCC)

---

## What this project asks

Omics datasets routinely measure far more features than they have samples, and the field still leans on classical linear pipelines — elastic net, PCA with a linear classifier, fixed signature scores. These are interpretable and reliable, but they assume broadly linear structure and often degrade when moved to a new cohort or batch.

This thesis asks not *whether* deep learning is better, but **under what conditions a learned representation actually beats those classical pipelines** — in accuracy, in robustness to degraded input, and in generalisation to an independent cohort.

Everything is benchmarked against a published method on its own data and labels: **MetaWIBELE** (Zhang et al., *Nature* 606:754–760, 2022), on the HMP2/IBDMDB inflammatory bowel disease cohort.

## Headline results

| Experiment | Finding |
|---|---|
| Sample-level dysbiosis classification | Classical models win on clean accuracy (ElasticNet AUROC 0.940); the autoencoder does not improve on them at this dimensionality |
| Robustness to corrupted input | Learned representations degrade far more gracefully; the VAE is most robust (0.892 at σ=3 noise vs 0.765 for ElasticNet) |
| Potentially bioactive protein-family ranking | Abundance MLP reaches AUPRC **0.142** vs MetaWIBELE's **0.076** (1.9×), winning in every annotation stratum |
| Sequence + multimodal fusion | ESM-2 650M sequence-only **0.271**; fusion with abundance **0.368** — 4.8× MetaWIBELE, 86 % precision in the top 100 |
| Cross-cohort generalisation | Autoencoder transfers at AUROC **0.790** vs ElasticNet **0.671** (HMP2 → PRISM), the clearest deep-learning win |

The overall conclusion is conditional: deep representation learning helps little at low dimensionality and small sample size, and substantially once the input is genuinely high-dimensional — most of all when ecological abundance and protein sequence are fused.

A methodological caution worth highlighting: naive sample-level cross-validation on the longitudinal cohort reports AUROC 0.95 for a task that is closer to **0.46** once whole patients are held out.

## Manuscript

| File | Contents |
|---|---|
| [`thesis.pdf`](thesis.pdf) | The manuscript (article format) |
| [`Supplementary_Material.pdf`](Supplementary_Material.pdf) | Supplementary Tables S1–S11, Figures S1–S15, and HPC compute notes |
| `thesis.md`, `supplementary.md` | Markdown sources |
| `build_pdf.py` | Builds `thesis.pdf` from `thesis.md` via pandoc + XeLaTeX |
| `draw_architecture.py` | Generates the network-architecture figure |

## Repository layout

```
data/
├── sample_classification/      Section 3: dysbiosis classification + robustness
│   ├── explore.ipynb          data prep, ElasticNet / PCA / AE comparison
│   ├── robust_vae.py          long-budget encoders (AE, DAE, VAE) + noise/missingness sweep
│   ├── figures/               generated figures
│   └── predictions/           result tables (CSV)
├── protein_prioritization/     Sections 4-5: protein-family ranking (main task)
│   ├── 01_data.ipynb          build 1,447,952 families, MPX positive labels
│   ├── 02_abundance_only.ipynb  linear + MLP on abundance profiles
│   ├── 03_sequence.ipynb      k-mer and ESM-2 sequence models, fusion
│   ├── esm_embed.py           ESM-2 embedding job (HPC, resumable, memory-mapped)
│   ├── 06_seeds.py            five-split multi-seed robustness
│   ├── bootstrap_ci.py        bootstrap confidence intervals on the fixed test set
│   ├── 07_complementarity.py  abundance vs sequence score independence
│   ├── interpret_taskb.py     Pfam / localization / MaAsLin2 enrichment of top families
│   ├── model_cards.py         parameter counts (Supplementary Table S1)
│   ├── run_esm_*.sh           Slurm submission scripts
│   ├── figures/
│   └── predictions/           result tables (CSV)
└── cross_cohort/               Section 6: cross-cohort generalisation
    ├── taskc_crosscohort.py   HMP2 <-> PRISM transfer on 197 shared species
    ├── ec_crosscohort.py      same on 2,052 shared EC enzymes
    ├── feature_importance.py  per-species Cohen's d, cross-cohort agreement
    ├── ordination.py          pooled PCA: batch effect vs diagnosis signal
    ├── figures/
    └── predictions/           result tables (CSV)
```

Result tables and figures are committed. Large inputs and intermediates are not — see below.

## Data

Raw data is **not** included in this repository (the protein-family inputs alone are ~2.7 GB). All of it is public:

- **HMP2 / IBDMDB** taxonomic, pathway and EC profiles, plus metadata: <https://ibdmdb.org/> (BioProject [PRJNA398089](https://www.ncbi.nlm.nih.gov/bioproject/398089))
- **MetaWIBELE outputs** — abundance matrix, representative sequences, cluster map, annotations, and both published priority tables — from the Huttenhower data portal, as released with Zhang et al. (2022)
- **Positive labels**: Supplementary Tables S7/S8 of Zhang et al. (2022), the metaproteomics-validated families
- **PRISM cohort** (external validation): Supplementary Datasets 4 and 6 of Franzosa et al. (2019)

Excluded from version control: `protein_prioritization/raw/`, all `.parquet` and `.pt` intermediates, compressed profile tables, and the third-party paper supplements.

## Reproducing

```bash
pip install -r requirements.txt
```

Then run, in order: `sample_classification/explore.ipynb` and `robust_vae.py`; `protein_prioritization/01_data.ipynb` → `02_abundance_only.ipynb` → `03_sequence.ipynb`; `cross_cohort/taskc_crosscohort.py` and `ec_crosscohort.py`.

ESM-2 embeddings for all 1.45 M families were computed on the CSUC Pirineus3 cluster (Slurm, NVIDIA H100) via `esm_embed.py`: 1 h 18 min for the 35M model, 4 h 32 min for the 650M model, producing a 7.4 GB matrix. Everything else runs on a laptop with the Apple-Silicon MPS backend.

All runs are deterministic — `torch.manual_seed(0)`, `np.random.seed(0)`, and a fixed scikit-learn `random_state` — and the same train/validation/test split is reused across the abundance, sequence and fusion experiments so every comparison is on identical test families. Every number and figure in the manuscript is produced by these scripts rather than entered by hand.

## Building the manuscript

```bash
python build_pdf.py                                    # -> thesis.pdf
pandoc supplementary.md -o Supplementary_Material.pdf --pdf-engine=xelatex
```

Requires pandoc and a XeLaTeX installation.

## References

Zhang Y, Bhosle A, Bae S, et al. Discovery of bioactive microbial gene products in inflammatory bowel disease. *Nature* 606:754–760 (2022).

Franzosa EA, Sirota-Madi A, Avila-Pacheco J, et al. Gut microbiome structure and metabolic activity in inflammatory bowel disease. *Nature Microbiology* 4:293–305 (2019).

Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379:1123–1130 (2023).

Full reference list in the manuscript.
