#!/usr/bin/env python
"""Precompute the v0 abundance MLP scores for the val+test families, so the 650M
fusion eval can run on the cluster WITHOUT uploading the 9 GB abundance matrix.

Reproduces the exact split and standardization of 02_abundance_only.ipynb /
03_sequence.ipynb, then writes processed/abund_scores_vate.parquet
(columns: family_id, split, abund_score).
"""
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split

PROC = Path('processed'); PRED = Path('predictions')
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

labels = pd.read_parquet(PROC / 'labels.parquet')
fids = pd.read_parquet(PROC / 'family_ids.parquet')['family_id'].values
assert (labels['family_id'].values == fids).all()
y_all = labels['is_bioactive'].astype(np.int8).values

idx = np.arange(len(y_all))
tr, te = train_test_split(idx, test_size=0.15, stratify=y_all, random_state=0)
tr, va = train_test_split(tr, test_size=0.15 / 0.85, stratify=y_all[tr], random_state=0)
tr_full = tr  # SMOKE=False

X = np.load(PROC / 'abundance.npy', mmap_mode='r')
n_feat = X.shape[1]
fsum = np.zeros(n_feat); fsq = np.zeros(n_feat); B = 16_384
tr_sorted = np.sort(tr_full)
for s in range(0, len(tr_sorted), B):
    rows = np.log1p(X[tr_sorted[s:s+B]].astype(np.float32))
    fsum += rows.sum(0); fsq += (rows**2).sum(0)
fmean = (fsum/len(tr_full)).astype(np.float32)
fstd = np.sqrt(np.maximum(fsq/len(tr_full) - fmean**2, 1e-12)).astype(np.float32)

class MLP(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Dropout(0.3),
                              nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))
    def forward(s,x): return s.net(x)

mlp_path = (PRED / '02_mlp.pt') if (PRED / '02_mlp.pt').exists() else (PROC / '02_mlp.pt')
m = MLP(n_feat).to(device); m.load_state_dict(torch.load(mlp_path, map_location=device)); m.eval()

def score(indices):
    out = np.empty(len(indices), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(indices), 4096):
            chunk = indices[s:s+4096]
            xb = np.log1p(X[np.sort(chunk)].astype(np.float32))
            order = np.argsort(chunk); inv = np.argsort(order)
            xb = (xb - fmean) / fstd
            sc = torch.sigmoid(m(torch.from_numpy(xb).to(device))).squeeze(-1).cpu().numpy()
            out[s:s+len(chunk)] = sc[inv]
    return out

rows = []
for nm, ix in [('val', va), ('test', te)]:
    sc = score(ix)
    rows.append(pd.DataFrame({'family_id': fids[ix], 'split': nm, 'abund_score': sc}))
out = pd.concat(rows, ignore_index=True)
out.to_parquet(PROC / 'abund_scores_vate.parquet', index=False)
print('wrote', PROC / 'abund_scores_vate.parquet', out.shape, 'val', (out.split=='val').sum(), 'test', (out.split=='test').sum())
