#!/usr/bin/env python
"""Biological interpretation of the Task B ranking: what is special about the
protein families the fusion model ranks at the top?

Using the MetaWIBELE functional annotations (extract_annotations.py), I compare
the top-1000 families by fusion score against the rest of the test universe and
test for enrichment of (1) Pfam domains, (2) PSORTb subcellular localization
(secreted / cell-wall / membrane vs cytoplasmic), (3) MaAsLin2 differential
abundance in dysbiosis, and (4) contributing genera. Fisher exact tests with
Benjamini-Hochberg FDR. Outputs predictions/interpret_*.csv and fig19.
"""
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.simplefilter('ignore')
from scipy.stats import fisher_exact
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

PFAM_NAMES = {
    'PF07715': 'TonB receptor, plug', 'PF00593': 'TonB receptor, barrel',
    'PF13715': 'SusC-assoc (CarboxypepD-reg)', 'PF14322': 'SusD-like (glycan binding)',
    'PF07980': 'SusD family', 'PF03144': 'EF-Tu domain 2',
    'PF00208': 'Glu/Leu/Phe/Val dehydrogenase', 'PF02812': 'ELFV dehydrogenase (N)',
    'PF14492': 'EF-G domain III', 'PF00009': 'EF-Tu GTP-binding',
    'PF03764': 'EF-G domain IV', 'PF02915': 'Rubrerythrin (oxidative stress)',
    'PF00679': 'EF-G C-terminus', 'PF01326': 'Pyruvate-P dikinase', 'PF02874': 'ATP synthase beta',
}

HERE = Path(__file__).parent; PRED = HERE / 'predictions'; FIG = HERE / 'figures'
ann = pd.read_parquet(PRED / 'fam_annotations.parquet')
pred = pd.read_parquet(PRED / '03_predictions_precomputed650.parquet')
df = pred.merge(ann, on='family_id', how='left')
N = len(df); K = 1000
df = df.sort_values('fusion_score', ascending=False).reset_index(drop=True)
df['top'] = np.arange(N) < K
print(f'universe {N}, top-K {K}, positives in top {int(df.top.sum() and df[df.top].y_true.sum())}', flush=True)

def bh(p):
    p = np.asarray(p); o = np.argsort(p); ranked = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]; out = np.empty_like(q); out[o] = q
    return np.clip(out, 0, 1)

def enrich(items_top, items_bg, min_count=20):
    """items_*: pd.Series of ';'-joined tokens. Fisher per token."""
    def counts(s):
        c = {}
        for v in s.dropna():
            for t in set(str(v).split(';')):
                if t: c[t] = c.get(t, 0) + 1
        return c
    ct, cb = counts(items_top), counts(items_bg)
    nt, nb = len(items_top), len(items_bg)
    rows = []
    for t, a in ct.items():
        tot = a + cb.get(t, 0)
        if tot < min_count: continue
        b = nt - a; c = cb.get(t, 0); d = nb - c
        orr, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        rows.append({'token': t, 'top_count': a, 'bg_count': c, 'odds_ratio': orr, 'p': p,
                     'top_frac': a/nt, 'bg_frac': c/nb})
    r = pd.DataFrame(rows)
    if len(r): r['fdr'] = bh(r['p'].values); r = r.sort_values('p')
    return r

top, bg = df[df.top], df[~df.top]

# (1) Pfam
pf = enrich(top['pfam'], bg['pfam'], min_count=25)
pf.to_csv(PRED / 'interpret_pfam.csv', index=False)
print('\nTop Pfam enrichments:\n', pf.head(15)[['token','top_count','bg_count','odds_ratio','fdr']].to_string(index=False), flush=True)

# (2) localization
loc_rows = []
for L in ['extracellular', 'cellwall', 'membrane', 'periplasmic', 'cytoplasmic']:
    a = (top['localization'] == L).sum(); c = (bg['localization'] == L).sum()
    b = len(top) - a; d = len(bg) - c
    orr, p = fisher_exact([[a, b], [c, d]], alternative='greater')
    loc_rows.append({'localization': L, 'top_frac': a/len(top), 'bg_frac': c/len(bg), 'odds_ratio': orr, 'p': p})
loc = pd.DataFrame(loc_rows); loc.to_csv(PRED / 'interpret_localization.csv', index=False)
print('\nLocalization enrichment:\n', loc.to_string(index=False), flush=True)

# (3) MaAsLin2 differential abundance (any call)
df['is_DA'] = df['maaslin'].notna()
a = df[df.top].is_DA.sum(); c = df[~df.top].is_DA.sum()
orr, p = fisher_exact([[a, len(top)-a], [c, len(bg)-c]], alternative='greater')
print(f'\nMaAsLin2 differential-abundance: top {a/len(top):.2f} vs bg {c/len(bg):.2f}  OR={orr:.2f} p={p:.1e}', flush=True)

# (4) genera
ge = enrich(top['genus'], bg['genus'], min_count=20)
ge.to_csv(PRED / 'interpret_genus.csv', index=False)
print('\nTop genera enriched:\n', ge.head(12)[['token','top_count','bg_count','odds_ratio','fdr']].to_string(index=False), flush=True)

# ---------- figure ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# localization panel
Ls = loc.set_index('localization').loc[['extracellular', 'cellwall', 'membrane', 'periplasmic', 'cytoplasmic']]
x = np.arange(len(Ls)); w = 0.38
axes[0].bar(x - w/2, Ls['top_frac'], w, label='top-1000 (predicted bioactive)', color='#c44e52', edgecolor='black', lw=0.5)
axes[0].bar(x + w/2, Ls['bg_frac'], w, label='background', color='#bdbdbd', edgecolor='black', lw=0.5)
axes[0].set_xticks(x); axes[0].set_xticklabels(Ls.index, rotation=20, ha='right')
axes[0].set_ylabel('fraction of families'); axes[0].set_title('Subcellular localization (PSORTb)')
axes[0].legend(fontsize=8)
# pfam panel
topf = pf.head(12).iloc[::-1]
lab = [f"{PFAM_NAMES.get(t, t)}" for t in topf['token']]
axes[1].barh(range(len(topf)), -np.log10(topf['fdr'] + 1e-300), color='#2e7d32', edgecolor='black', lw=0.4)
axes[1].set_yticks(range(len(topf))); axes[1].set_yticklabels(lab, fontsize=8)
axes[1].set_xlabel('-log10 FDR (enrichment in top-1000)')
axes[1].set_title('Enriched Pfam domains')
plt.tight_layout(); plt.savefig(FIG / 'fig19_interpret_taskb.png', dpi=150); plt.close()
print('wrote figures/fig19_interpret_taskb.png', flush=True)
