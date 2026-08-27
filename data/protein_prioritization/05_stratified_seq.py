#!/usr/bin/env python
"""Stratified analysis of the Task B v1 models (ESM-2 650M sequence-only and
abundance+sequence fusion) by protein-characterization bucket, alongside the
abundance MLP and the published MetaWIBELE scores.

Question: does the sequence model rescue the weak-homology and novel families
where the abundance model was weakest (Section 4.3, Table 3)?

Uses only saved artifacts (no retraining):
  predictions/03_predictions_precomputed650.parquet  (family_id, y_true, seq_score, fusion_score)
  processed/abund_scores_vate.parquet                (test abundance MLP scores)
  processed/labels.parquet                           (MetaWIBELE ranks)
  processed/annotations.parquet                      (homology -> bucket)
Outputs predictions/05_stratified_seq.csv and figures/fig10_stratified_seq.png.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

PROC = Path('processed'); PRED = Path('predictions'); FIG = Path('figures')

pred = pd.read_parquet(PRED / '03_predictions_precomputed650.parquet')  # test families
ab = pd.read_parquet(PROC / 'abund_scores_vate.parquet')
ab = ab[ab['split'] == 'test'][['family_id', 'abund_score']]
lab = pd.read_parquet(PROC / 'labels.parquet')[['family_id', 'metawibele_unsup_rank', 'metawibele_sup_rank']]
ann = pd.read_parquet(PROC / 'annotations.parquet')

df = pred.merge(ab, on='family_id', how='left').merge(lab, on='family_id', how='left').merge(ann, on='family_id', how='left')

def bucket(r):
    if isinstance(r['strong_homology'], str) and r['strong_homology']:
        return 'characterized'
    if isinstance(r['weak_homology'], str) and r['weak_homology']:
        return 'weak'
    return 'novel'
df['bucket'] = df.apply(bucket, axis=1)

methods = {
    'MetaWIBELE unsup': 'metawibele_unsup_rank',
    'MetaWIBELE sup':   'metawibele_sup_rank',
    'Abundance MLP':    'abund_score',
    'Sequence ESM-2 650M': 'seq_score',
    'Fusion (abund+ESM-2)': 'fusion_score',
}
rows = []
for b in ['characterized', 'weak', 'novel']:
    sub = df[df['bucket'] == b]
    y = sub['y_true'].values
    rec = {'bucket': b, 'n': len(sub), 'pos': int(y.sum()), 'base_rate': round(y.mean(), 4)}
    for name, col in methods.items():
        s = sub[col].fillna(0).values
        rec[name] = round(average_precision_score(y, s), 4) if y.sum() > 0 else float('nan')
    rows.append(rec)
res = pd.DataFrame(rows)
res.to_csv(PRED / '05_stratified_seq.csv', index=False)
print(res.to_string(index=False))

# figure: grouped bars per bucket
buckets = res['bucket'].tolist()
plot_methods = ['MetaWIBELE unsup', 'Abundance MLP', 'Sequence ESM-2 650M', 'Fusion (abund+ESM-2)']
colors = ['#9e9e9e', '#4c72b0', '#2e7d32', '#c44e52']
x = np.arange(len(buckets)); w = 0.2
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for i, (m, c) in enumerate(zip(plot_methods, colors)):
    ax.bar(x + (i - 1.5) * w, res[m].values, w, label=m, color=c, edgecolor='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{b}\n(n={n:,}, pos={p})" for b, n, p in zip(buckets, res['n'], res['pos'])])
ax.set_ylabel('AUPRC (test set)')
ax.set_title('AUPRC by protein-characterization bucket')
ax.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(FIG / 'fig10_stratified_seq.png', dpi=150); plt.close()
print('wrote figures/fig10_stratified_seq.png')
