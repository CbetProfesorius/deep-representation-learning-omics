#!/usr/bin/env python
"""fig9: consolidated comparison of our models against the published MetaWIBELE
baseline (AUPRC), with relative-improvement multipliers annotated."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = Path('figures')
mw = 0.076  # MetaWIBELE unsupervised (published), our test families

models = ['MetaWIBELE\nunsup\n(published)', 'Linear\n(ours)', 'Abundance\nMLP (ours)',
          'ESM-2 650M\nsequence', 'Fusion\nabund+ESM-2']
auprc  = [0.076, 0.104, 0.142, 0.271, 0.368]
colors = ['#9e9e9e', '#7a9cc6', '#4c72b0', '#2e7d32', '#c44e52']

fig, ax = plt.subplots(figsize=(8.6, 4.8))
bars = ax.bar(models, auprc, color=colors, edgecolor='black', lw=0.6)
ax.axhline(mw, ls='--', c='gray', lw=1)
for b, v in zip(bars, auprc):
    mult = v / mw
    tag = 'baseline' if abs(v - mw) < 1e-9 else f'{v:.3f}\n({mult:.1f}x)'
    ax.text(b.get_x() + b.get_width()/2, v + 0.006, tag, ha='center', va='bottom', fontsize=9)
ax.set_ylabel('AUPRC (test set, 217,193 families)')
ax.set_title('Improvement over the published MetaWIBELE baseline (Zhang et al. 2022)')
ax.set_ylim(0, 0.42)
plt.tight_layout(); plt.savefig(FIG / 'fig9_vs_metawibele.png', dpi=150); plt.close()
print('wrote figures/fig9_vs_metawibele.png')
