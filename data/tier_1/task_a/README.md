# Task A — Sample-level phenotype classification

## What's in this folder

| File | Size | Shape (rows × cols) | What it is |
|---|---|---|---|
| `hmp2_metadata.csv` | 8.7 MB | 5,534 rows × 489 cols | Per-sample metadata: patient ID, week, diagnosis (CD/UC/non-IBD), data type, antibiotics, demographics, etc. |
| `dysbiosis_scores.tsv` | 44 KB | 1,594 rows × 3 cols | `sample_id <TAB> dysbiosis_score <TAB> is_dysbiotic` |
| `taxonomic_profiles.tsv.gz` | 651 KB | 1,480 features × 1,638 samples | MetaPhlAn relative abundances of microbial taxa (kingdom → species). First column = taxon name |
| `pathabundance_relab.tsv.gz` | 8.6 MB | 10,885 features × 1,638 samples | HUMAnN MetaCyc pathway relative abundances, both unstratified and species-stratified |
| `ecs_relab.tsv.gz` | 85 MB | 108,434 features × 1,638 samples | HUMAnN Enzyme Commission number relative abundances, both unstratified and species-stratified |

All matrices are **features × samples** (the columns after the first are sample IDs).

## Important notes about the matrices

The HUMAnN tables (`pathabundance_relab`, `ecs_relab`) contain **two kinds of rows mixed together**:

- **Community-level rows** (e.g., `1CMET2-PWY: N10-formyl-tetrahydrofolate biosynthesis`) — total abundance of that pathway in the community.
- **Stratified rows** (e.g., `1CMET2-PWY: ... |g__Bacteroides.s__Bacteroides_caccae`) — abundance of that pathway contributed by one specific species.

For the first model, **use only the community-level rows** (rows that don't contain `|`). The stratified rows are useful later for taxonomy-aware modelling (Task B / D).

## Sources

- **Metadata** — [`hmp2_metadata_2018-08-20.csv`](https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/metadata/hmp2_metadata_2018-08-20.csv)
- **Dysbiosis scores** — [bioBakery forum link](https://forum.biobakery.org/uploads/short-url/umwfR0kDJ6s5RXHwtIMgLaOEoOI.tsv)
- **Merged HUMAnN tables** — [`ibdmdb.org/results`](https://ibdmdb.org/results) → MGX products → [`products_MGX_2017-08-12.html`](https://ibdmdb.org/downloads/html/products_MGX_2017-08-12.html), folder `products/HMP2/MGX/2018-05-04/`

## Suggested order of use

1. Start with `taxonomic_profiles.tsv.gz` — smallest, easiest to load, ~1,500 features. Get the whole pipeline working end-to-end here.
2. Move to `pathabundance_relab.tsv.gz` (community rows only, ~500 features) — same shape as #1.
3. Scale up to `ecs_relab.tsv.gz` (community rows only, ~3,000 features). This is your high-dimensional input.
4. (Later) `genefamilies_relab.tsv.gz` — multi-GB, millions of features — only after the rest works.

## Label sources for classification

- **Dysbiotic vs. non-dysbiotic** — column `is_dysbiotic` in `dysbiosis_scores.tsv`. This is the supervised label used by the Zhang 2022 paper.
- **CD vs. UC vs. non-IBD** — column `diagnosis` in `hmp2_metadata.csv`.
- **Continuous dysbiosis score** — column 2 of `dysbiosis_scores.tsv`, for regression.

Always join on the sample ID (the column header in the feature matrices == the `External ID` / sample ID in metadata).

## Critical evaluation rule

**Split by `Participant ID`, not by sample.** Each patient has many samples over time. Putting samples from the same patient in both train and test leaks information and inflates AUROC.
