#!/usr/bin/env python
"""Pull the MetaWIBELE functional annotations for the families I scored, so the
top-ranked predictions can be interpreted biologically. One streaming pass over
the 707 MB annotation table; keep only families in the test universe and only the
fields useful for interpretation: assigned species/taxonomy, Pfam domains,
PSORTb subcellular localization, and MaAsLin2 differential-abundance calls.
Writes predictions/fam_annotations.parquet.
"""
import gzip
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / 'raw' / 'HMP2_proteinfamilies_annotation.tsv.gz'
PRED = HERE / 'predictions'

universe = set(pd.read_parquet(PRED / '03_predictions_precomputed650.parquet')['family_id'])
print('test universe:', len(universe), flush=True)

species, pfam, loc, maaslin = {}, {}, {}, {}
PSORT = {'PSORTb_cytoplasmic': 'cytoplasmic', 'PSORTb_cytoplasmicMembrane': 'membrane',
         'PSORTb_cellwall': 'cellwall', 'PSORTb_extracellular': 'extracellular',
         'PSORTb_periplasmic': 'periplasmic', 'PSORTb_outerMembrane': 'outerMembrane'}
n = 0
with gzip.open(RAW, 'rt') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) < 5:
            continue
        fam, ann, feat = p[0], p[1], p[2]
        if fam not in universe:
            continue
        if feat == 'Species':
            species[fam] = ann
        elif feat == 'InterProScan_PfamDomain':
            pfam[fam] = ann  # 'PF00587;PF02403'
        elif feat in PSORT:
            loc[fam] = PSORT[feat]
        elif feat == 'MaAsLin2_DA':
            maaslin.setdefault(fam, []).append(ann)
        n += 1
print('matched annotation rows:', n, flush=True)

fams = sorted(universe)
df = pd.DataFrame({'family_id': fams})
df['species'] = df['family_id'].map(species)
df['pfam'] = df['family_id'].map(pfam)
df['localization'] = df['family_id'].map(loc)
df['maaslin'] = df['family_id'].map(lambda f: ';'.join(maaslin.get(f, [])) or None)
df['genus'] = df['species'].str.split().str[0]
df.to_parquet(PRED / 'fam_annotations.parquet', index=False)
print('wrote predictions/fam_annotations.parquet', df.shape, flush=True)
print('coverage: species %.2f, pfam %.2f, loc %.2f, maaslin %.2f' % (
    df['species'].notna().mean(), df['pfam'].notna().mean(),
    df['localization'].notna().mean(), df['maaslin'].notna().mean()), flush=True)
