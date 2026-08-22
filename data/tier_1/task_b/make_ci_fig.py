#!/usr/bin/env python
"""fig12: multi-seed (n=5) test AUPRC with 95% CI error bars per method,
showing the gaps over MetaWIBELE are not seed noise."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRED = Path('predictions'); FIG = Path('figures')
s = pd.read_csv(PRED / '06_seed_summary.csv', index_col=0)
order = ['MetaWIBELE sup', 'MetaWIBELE unsup', 'Abundance MLP', 'Sequence ESM-2 35M', 'Fusion']
s = s.loc[order]
colors = ['#bdbdbd', '#9e9e9e', '#4c72b0', '#2e7d32', '#c44e52']
err = np.c_[s['mean'] - s['ci95_lo'], s['ci95_hi'] - s['mean']].T

fig, ax = plt.subplots(figsize=(8.4, 4.6))
bars = ax.bar(range(len(s)), s['mean'], yerr=err, capsize=5,
              color=colors, edgecolor='black', lw=0.6)
ax.set_xticks(range(len(s)))
ax.set_xticklabels([m.replace(' ', '\n', 1) for m in order])
for i, (m, lo, hi) in enumerate(zip(s['mean'], s['ci95_lo'], s['ci95_hi'])):
    ax.text(i, hi + 0.006, f'{m:.3f}', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('test AUPRC (mean of 5 seeds, 95% CI)')
ax.set_title('Robustness across 5 random splits/seeds (ESM-2 35M pipeline)')
ax.set_ylim(0, 0.40)
plt.tight_layout(); plt.savefig(FIG / 'fig12_seed_ci.png', dpi=150); plt.close()
print('wrote figures/fig12_seed_ci.png')
