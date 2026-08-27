#!/usr/bin/env python
"""Bootstrap 95% confidence intervals for the Task B headline metrics.

The multi-seed CIs in Section 5.5 capture variation from re-training on different
splits (35M pipeline). This complements them with sampling error on the fixed
650M test set: resample the 217,193 test families with replacement (B=1000) and
recompute AUPRC/AUROC, giving a percentile CI per method. Uses only saved
predictions. Outputs predictions/bootstrap_ci.csv and figures/fig20_bootstrap_ci.png.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

PRED = Path('predictions'); FIG = Path('figures')
a = pd.read_parquet(PRED / '02_abundance_only.parquet')
b = pd.read_parquet(PRED / '03_predictions_precomputed650.parquet')[['family_id', 'seq_score', 'fusion_score']]
df = a.merge(b, on='family_id', how='inner')
y = df['y_true'].values.astype(int)
methods = {'MetaWIBELE unsup': 'mw_unsup', 'MetaWIBELE sup': 'mw_sup',
           'Abundance MLP': 'mlp_score', 'Sequence ESM-2 650M': 'seq_score',
           'Fusion (abund+ESM-2)': 'fusion_score'}
print('n test', len(df), 'pos', int(y.sum()), flush=True)

B = 1000
rng = np.random.default_rng(0)
N = len(y)
idx_boot = [rng.integers(0, N, N) for _ in range(B)]

rows = []
for name, col in methods.items():
    s = df[col].fillna(0).values
    ap0, au0 = average_precision_score(y, s), roc_auc_score(y, s)
    aps, aus = np.empty(B), np.empty(B)
    for i, ix in enumerate(idx_boot):
        yi = y[ix]
        if yi.sum() == 0 or yi.sum() == len(yi):
            aps[i] = np.nan; aus[i] = np.nan; continue
        aps[i] = average_precision_score(yi, s[ix]); aus[i] = roc_auc_score(yi, s[ix])
    rows.append({'method': name, 'auprc': round(ap0, 4),
                 'auprc_lo': round(np.nanpercentile(aps, 2.5), 4), 'auprc_hi': round(np.nanpercentile(aps, 97.5), 4),
                 'auroc': round(au0, 4),
                 'auroc_lo': round(np.nanpercentile(aus, 2.5), 4), 'auroc_hi': round(np.nanpercentile(aus, 97.5), 4)})
    print(f"{name}: AUPRC {ap0:.3f} [{rows[-1]['auprc_lo']:.3f}, {rows[-1]['auprc_hi']:.3f}] | "
          f"AUROC {au0:.3f} [{rows[-1]['auroc_lo']:.3f}, {rows[-1]['auroc_hi']:.3f}]", flush=True)

res = pd.DataFrame(rows)
res.to_csv(PRED / 'bootstrap_ci.csv', index=False)

# figure: AUPRC with bootstrap CIs
colors = ['#9e9e9e', '#bdbdbd', '#4c72b0', '#2e7d32', '#c44e52']
err = np.c_[res['auprc'] - res['auprc_lo'], res['auprc_hi'] - res['auprc']].T
fig, ax = plt.subplots(figsize=(8.4, 4.7))
ax.bar(range(len(res)), res['auprc'], yerr=err, capsize=5, color=colors, edgecolor='black', lw=0.6)
ax.set_xticks(range(len(res))); ax.set_xticklabels([m.replace(' ', '\n', 1) for m in methods], fontsize=8)
for i, r in res.iterrows():
    ax.text(i, r['auprc_hi'] + 0.006, f"{r['auprc']:.3f}", ha='center', fontsize=8)
ax.set_ylabel('test AUPRC (650M test set, 95% bootstrap CI)')
ax.set_title('Bootstrap confidence intervals on the test set (B=1000)')
ax.set_ylim(0, 0.42)
plt.tight_layout(); plt.savefig(FIG / 'fig20_bootstrap_ci.png', dpi=150); plt.close()
print('wrote figures/fig20_bootstrap_ci.png', flush=True)
