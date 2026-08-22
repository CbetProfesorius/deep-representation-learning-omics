#!/usr/bin/env python
"""Why does fusion beat either signal alone? Quantify how complementary the
abundance and sequence (ESM-2 650M) rankings are on the test set:
  - Spearman correlation between the two scores (low => independent signals),
  - top-K set overlap (Jaccard) of the two rankings,
  - of the true positives recovered in each model's top-1000, how many are
    unique to abundance vs unique to sequence vs shared.
Uses saved artifacts only. Writes predictions/07_complementarity.csv and
figures/fig11_complementarity.png.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PROC = Path('processed'); PRED = Path('predictions'); FIG = Path('figures')

pred = pd.read_parquet(PRED / '03_predictions_precomputed650.parquet')
ab = pd.read_parquet(PROC / 'abund_scores_vate.parquet')
ab = ab[ab['split'] == 'test'][['family_id', 'abund_score']]
df = pred.merge(ab, on='family_id', how='left').dropna(subset=['abund_score'])
y = df['y_true'].values
a = df['abund_score'].values
s = df['seq_score'].values

rho, _ = spearmanr(a, s)
print(f'Spearman(abundance, sequence) = {rho:.3f}  (n={len(df):,})')

def topk_set(score, k): return set(np.argsort(-score)[:k])
out_rows = []
for k in (100, 1000, 5000):
    A, S = topk_set(a, k), topk_set(s, k)
    jacc = len(A & S) / len(A | S)
    # true positives recovered in each top-k
    pos = set(np.where(y == 1)[0])
    a_hits, s_hits = A & pos, S & pos
    shared = a_hits & s_hits
    out_rows.append({'k': k, 'jaccard_topk': round(jacc, 3),
                     'tp_abund': len(a_hits), 'tp_seq': len(s_hits),
                     'tp_shared': len(shared),
                     'tp_union': len(a_hits | s_hits)})
res = pd.DataFrame(out_rows)
res.insert(0, 'spearman', round(rho, 3))
res.to_csv(PRED / '07_complementarity.csv', index=False)
print(res.to_string(index=False))

# figure: hexbin of the two score ranks + TP-recovery bars at k=1000
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
ar = pd.Series(a).rank(pct=True).values
sr = pd.Series(s).rank(pct=True).values
hb = ax.hexbin(ar, sr, gridsize=45, bins='log', cmap='viridis')
ax.set_xlabel('abundance score (percentile rank)'); ax.set_ylabel('sequence score (percentile rank)')
ax.set_title(f'Score-rank density (Spearman {rho:.2f})')
fig.colorbar(hb, ax=ax, label='log10(count)')

ax = axes[1]
k = 1000
A, S = topk_set(a, k), topk_set(s, k)
pos = set(np.where(y == 1)[0])
a_only = len((A & pos) - S); s_only = len((S & pos) - A); shared = len(A & S & pos)
ax.bar(['abundance\nonly', 'shared', 'sequence\nonly'], [a_only, shared, s_only],
       color=['#4c72b0', '#9e9e9e', '#2e7d32'], edgecolor='black', lw=0.6)
for i, v in enumerate([a_only, shared, s_only]):
    ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)
ax.set_ylabel('true bioactive families recovered')
ax.set_title(f'Positives in top-{k}: where do hits come from?')
plt.tight_layout(); plt.savefig(FIG / 'fig11_complementarity.png', dpi=150); plt.close()
print('wrote figures/fig11_complementarity.png')
