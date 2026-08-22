#!/usr/bin/env python
"""Regenerate Task B v1 figures with ESM-2 (35M) results alongside the k-mer baseline.

fig6_v1_auprc.png : AUPRC progression MetaWIBELE -> abundance -> sequence (k-mer, ESM) -> fusion.
fig7_v1_pr.png    : precision-recall curves for the ESM-2 sequence-only and fusion models vs baselines.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

PRED = Path('predictions'); FIG = Path('figures'); FIG.mkdir(exist_ok=True)

kmer = pd.read_csv(PRED / '03_results_kmer.csv').set_index('method')
esm  = pd.read_csv(PRED / '03_results_precomputed650.csv').set_index('method')
base_rate = 0.0187

# ---- fig6: AUPRC progression ----
labels = ['MetaWIBELE\n(unsup)', 'Abundance\nMLP (v0)', 'Sequence\nk-mer', 'Sequence\nESM-2 650M',
          'Fusion\nabund+k-mer', 'Fusion\nabund+ESM-2']
vals = [esm.loc['MetaWIBELE unsup', 'auprc'],
        esm.loc['Abundance v0 (MLP)', 'auprc'],
        kmer.loc['Sequence-only (kmer)', 'auprc'],
        esm.loc['Sequence-only (ESM-2 650M)', 'auprc'],
        kmer.loc['Fusion [val-fitted logreg]', 'auprc'],
        esm.loc['Fusion (abundance + ESM-2 650M)', 'auprc']]
colors = ['#9e9e9e', '#4c72b0', '#88b04b', '#2e7d32', '#dd8452', '#c44e52']
fig, ax = plt.subplots(figsize=(8.5, 4.8))
bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.6)
ax.axhline(base_rate, ls='--', c='red', lw=1, label=f'random base rate ({base_rate:.3f})')
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.006, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('AUPRC (test set)')
ax.set_title('Bioactivity ranking: sequence signal beyond abundance')
ax.set_ylim(0, max(vals) * 1.18)
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout(); plt.savefig(FIG / 'fig6_v1_auprc.png', dpi=150); plt.close()
print('wrote fig6_v1_auprc.png', [round(v, 4) for v in vals])

# ---- fig7: PR curves (ESM-2 650M) ----
dfe = pd.read_parquet(PRED / '03_predictions_precomputed650.parquet')
y = dfe['y_true'].values
fig, ax = plt.subplots(figsize=(6.8, 5.2))
def pr(y, s, label, **kw):
    p, r, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    ax.plot(r, p, label=f'{label} (AP={ap:.3f})', **kw)
pr(y, dfe['fusion_score'].values, 'Fusion (abund+ESM-2 650M)', color='#c44e52', lw=2.2)
pr(y, dfe['seq_score'].values,    'ESM-2 650M sequence-only',  color='#2e7d32', lw=2)
# baselines on the same test rows
v0 = pd.read_parquet(PRED / '02_abundance_only.parquet')
m = dict(zip(v0['family_id'], v0['mlp_score']))
abund = np.array([m.get(f, np.nan) for f in dfe['family_id']], dtype=float)
ok = ~np.isnan(abund)
pr(y[ok], abund[ok], 'Abundance MLP (v0)', color='#4c72b0', lw=1.6, ls='--')
ax.axhline(base_rate, ls=':', c='gray', lw=1, label=f'random ({base_rate:.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision–recall (ESM-2 650M)')
ax.legend(loc='upper right', fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(FIG / 'fig7_v1_pr.png', dpi=150); plt.close()
print('wrote fig7_v1_pr.png')
