#!/usr/bin/env python
"""Ordination (PCA) of the two cohorts in the shared 197-species space.

Two questions an ordination answers cheaply: (1) how strong is the cohort/batch
effect (do HMP2 and Franzosa separate just by study?), and (2) is there visible
IBD-vs-control structure that a classifier could pick up? I project both cohorts
into the same PCA space (fit on the pooled, standardized log-abundances) and
colour by cohort and by diagnosis. I also quantify each axis of structure by how
well a logistic model on the first 10 PCs recovers cohort vs diagnosis.
Outputs predictions/ordination_variance.csv and figures/fig21_ordination.png.
"""
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.simplefilter('ignore')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).parent; TA = HERE.parent / 'sample_classification'; PRED = HERE / 'predictions'; FIG = HERE / 'figures'
hmp = pd.read_parquet(TA / 'processed' / 'X_tax_species.parquet')
lab = pd.read_parquet(TA / 'processed' / 'labels.parquet').set_index('sample_id').loc[hmp.index]
y_h = lab['diagnosis'].isin(['CD', 'UC']).astype(int).values
fr_raw = pd.read_excel(HERE / 'raw' / 'moesm6.xlsx', sheet_name='dataset_s4', header=1)
fr_raw = fr_raw.rename(columns={fr_raw.columns[0]: 'feature'})
META = ['SRA_metagenome_name', 'Age', 'Diagnosis', 'Fecal.Calprotectin', 'antibiotic', 'immunosuppressant', 'mesalamine', 'steroids']
y_f = fr_raw[fr_raw['feature'] == 'Diagnosis'].iloc[0, 1:].isin(['CD', 'UC']).astype(int).values
fr = fr_raw[~fr_raw['feature'].isin(META)].set_index('feature').apply(pd.to_numeric, errors='coerce').T.fillna(0.0)
shared = sorted(set(hmp.columns) & set(fr.columns))

def prep(df):
    X = df.reindex(columns=shared).fillna(0.0).values.astype(np.float64)
    X = X / np.clip(X.sum(1, keepdims=True), 1e-9, None)
    return np.log10(X + 1e-5)
Xh, Xf = prep(hmp), prep(fr)
X = np.vstack([Xh, Xf])
cohort = np.r_[np.zeros(len(Xh)), np.ones(len(Xf))]   # 0=HMP2, 1=Franzosa
ibd = np.r_[y_h, y_f]

Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=10, random_state=0).fit(Xs)
Z = pca.transform(Xs)
ev = pca.explained_variance_ratio_
print('variance explained PC1..PC5:', np.round(ev[:5], 3), flush=True)

# quantify structure: 5-fold CV AUROC of logistic on 10 PCs for cohort vs diagnosis
auc_cohort = cross_val_score(LogisticRegression(max_iter=2000), Z, cohort, cv=5, scoring='roc_auc').mean()
auc_ibd = cross_val_score(LogisticRegression(max_iter=2000, class_weight='balanced'), Z, ibd, cv=5, scoring='roc_auc').mean()
pd.DataFrame({'pc': range(1, 11), 'explained_variance_ratio': ev}).to_csv(PRED / 'ordination_variance.csv', index=False)
print(f'10-PC logistic AUROC: cohort {auc_cohort:.3f} | diagnosis {auc_ibd:.3f}', flush=True)

# ---------- figure ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
ax = axes[0]
for c, col, lbl in [(0, '#4c72b0', 'HMP2'), (1, '#dd8452', 'Franzosa')]:
    m = cohort == c
    ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.45, color=col, label=lbl)
ax.set_title(f'Coloured by cohort (10-PC separability AUROC {auc_cohort:.2f})')
ax.legend(fontsize=9)
ax = axes[1]
for v, col, lbl in [(0, '#2e7d32', 'control'), (1, '#c44e52', 'IBD (CD/UC)')]:
    m = ibd == v
    ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.45, color=col, label=lbl)
ax.set_title(f'Coloured by diagnosis (10-PC separability AUROC {auc_ibd:.2f})')
ax.legend(fontsize=9)
for ax in axes:
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% var)'); ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}% var)')
fig.suptitle('PCA ordination of HMP2 + Franzosa in the shared 197-species space', y=1.02, fontsize=12)
plt.tight_layout(); plt.savefig(FIG / 'fig21_ordination.png', dpi=150, bbox_inches='tight'); plt.close()
print('wrote figures/fig21_ordination.png', flush=True)
