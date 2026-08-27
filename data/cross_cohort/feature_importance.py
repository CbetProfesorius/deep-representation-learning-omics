#!/usr/bin/env python
"""Biological interpretation of the cross-cohort classifier: which species drive
the IBD-vs-control prediction, do the two cohorts agree on them, and do they
match the known IBD microbiome signature?

I fit an interpretable ElasticNet logistic model on the shared 197-species space
separately in HMP2 and in Franzosa, then compare the signed coefficients (a
positive weight pushes a sample toward IBD). Concordant, biologically sensible
weights are the mechanistic reason a model trained on one cohort transfers to
the other. Outputs predictions/feature_importance.csv and fig18.
"""
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.simplefilter('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).parent; TA = HERE.parent / 'sample_classification'
PRED = HERE / 'predictions'; FIG = HERE / 'figures'; PRED.mkdir(exist_ok=True)

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

def effect(X, y):
    """Standardized differential abundance per species: mean(log-abund | IBD) -
    mean(| control), scaled by pooled SD (Cohen's d). Stable and interpretable."""
    pos, neg = X[y == 1], X[y == 0]
    mp, mn = pos.mean(0), neg.mean(0)
    sd = np.sqrt(((pos.var(0) * len(pos) + neg.var(0) * len(neg)) / (len(X))) + 1e-9)
    return (mp - mn) / sd

ch, cf = effect(prep(hmp), y_h), effect(prep(fr), y_f)
imp = pd.DataFrame({'species': shared, 'hmp2_coef': ch, 'franzosa_coef': cf})
imp['mean_coef'] = imp[['hmp2_coef', 'franzosa_coef']].mean(1)
imp = imp.sort_values('mean_coef')
imp.to_csv(PRED / 'feature_importance.csv', index=False)

# known IBD signature (well established in the literature)
ibd_up = ['Escherichia_coli', 'Ruminococcus_gnavus', 'Klebsiella', 'Veillonella', 'Fusobacterium',
          'Clostridium_bolteae', 'Clostridium_clostridioforme', 'Clostridium_symbiosum', 'Proteobacteria', 'Enterococcus']
ibd_down = ['Faecalibacterium_prausnitzii', 'Roseburia', 'Eubacterium_rectale', 'Subdoligranulum',
            'Ruminococcus_bromii', 'Bifidobacterium', 'Eubacterium_hallii', 'Coprococcus', 'Alistipes', 'Akkermansia']
def known(s):
    if any(k in s for k in ibd_up): return 'up'
    if any(k in s for k in ibd_down): return 'down'
    return None
imp['known_ibd'] = imp['species'].map(known)

r, _ = pearsonr(ch, cf)
# concordance among species with a known direction
kn = imp.dropna(subset=['known_ibd'])
agree = ((kn['mean_coef'] > 0) == (kn['known_ibd'] == 'up')).mean()
print(f'cross-cohort coef correlation r={r:.3f}; known-signature species n={len(kn)}, '
      f'direction agreement={agree:.2f}', flush=True)
print('\nTop IBD-associated (positive):')
print(imp.tail(12)[['species', 'hmp2_coef', 'franzosa_coef', 'known_ibd']].iloc[::-1].to_string(index=False))
print('\nTop control-associated (negative):')
print(imp.head(12)[['species', 'hmp2_coef', 'franzosa_coef', 'known_ibd']].to_string(index=False))

# ---------- figure: panel A top species, panel B cross-cohort scatter ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
top = pd.concat([imp.head(12), imp.tail(12)])
colors = ['#c44e52' if c > 0 else '#4c72b0' for c in top['mean_coef']]
axes[0].barh(range(len(top)), top['mean_coef'], color=colors, edgecolor='black', lw=0.4)
axes[0].set_yticks(range(len(top)))
labels = [s.replace('_', ' ') + ('  *' if k in ('up', 'down') else '') for s, k in zip(top['species'], top['known_ibd'])]
axes[0].set_yticklabels(labels, fontsize=7.5)
axes[0].axvline(0, c='gray', lw=0.8)
axes[0].set_xlabel("standardized IBD-control difference (Cohen's d; +IBD / -control)")
axes[0].set_title('Top differentially abundant species (IBD vs control)\n(* = matches known IBD signature)')

axes[1].scatter(ch, cf, s=14, alpha=0.5, color='#777777')
for _, row in kn.iterrows():
    axes[1].scatter(row['hmp2_coef'], row['franzosa_coef'],
                    color='#c44e52' if row['known_ibd'] == 'up' else '#4c72b0', s=28, edgecolor='black', lw=0.4)
axes[1].axhline(0, c='gray', lw=0.6); axes[1].axvline(0, c='gray', lw=0.6)
axes[1].set_xlabel('HMP2 effect (d)'); axes[1].set_ylabel('Franzosa effect (d)')
axes[1].set_title(f'Cross-cohort agreement of species effects (r = {r:.2f})')
plt.tight_layout(); plt.savefig(FIG / 'fig18_feature_importance.png', dpi=150); plt.close()
print('wrote figures/fig18_feature_importance.png', flush=True)
