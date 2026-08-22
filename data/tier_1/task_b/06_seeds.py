#!/usr/bin/env python
"""Multi-seed confidence intervals for Task B.

For each of N seeds: redo the stratified 70/15/15 split, retrain the abundance
MLP and the ESM-2 35M sequence MLP, fit the val-based fusion, and score every
method (incl. the fixed published MetaWIBELE priorities) on that seed's test
split. Report mean +/- std (and a 95% normal CI) of AUPRC per method, so the
"beats MetaWIBELE" claim comes with error bars.

Reuses processed/seq_emb_35M.npy (per-family, split-independent) so only the
small heads are retrained. Writes predictions/06_seed_results.csv (per seed)
and predictions/06_seed_summary.csv (mean/std/CI).
"""
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, time
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

PROC = Path('processed'); PRED = Path('predictions')
SEEDS = [0, 1, 2, 3, 4]
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device, flush=True)

labels = pd.read_parquet(PROC / 'labels.parquet')
fids = pd.read_parquet(PROC / 'family_ids.parquet')['family_id'].values
assert (labels['family_id'].values == fids).all()
y_all = labels['is_bioactive'].astype(np.int8).values
mw_u_all = labels['metawibele_unsup_rank'].fillna(0).values
mw_s_all = labels['metawibele_sup_rank'].fillna(0).values

print('loading abundance into RAM...', flush=True); t0 = time.time()
X = np.load(PROC / 'abundance.npy')           # ~9.2 GB float32 in RAM
E = np.load(PROC / 'seq_emb_35M.npy', mmap_mode='r')  # 480-dim, keep on disk
print(f'  abundance {X.shape} loaded in {time.time()-t0:.0f}s; emb {E.shape}', flush=True)

def mlp(d):
    return nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))

def train_head(feat_fn, dim, tr, va, epochs=8):
    """feat_fn(idx)->standardized float32 array; trains an MLP head, early-stops on val AUPRC."""
    # train standardization
    Xtr = feat_fn(tr, raw=True)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    del Xtr
    model = mlp(dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pw = torch.tensor([(len(tr)-y_all[tr].sum())/max(y_all[tr].sum(),1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    def batches(idx, bs, shuffle):
        order = np.random.permutation(len(idx)) if shuffle else np.arange(len(idx))
        for s in range(0, len(idx), bs):
            chunk = idx[order[s:s+bs]]
            xb = (feat_fn(chunk, raw=True) - mu) / sd
            yield torch.from_numpy(xb.astype(np.float32)), torch.from_numpy(y_all[chunk].astype(np.float32))
    def predict(idx):
        model.eval(); out=[]
        with torch.no_grad():
            for s in range(0, len(idx), 8192):
                chunk = idx[s:s+8192]
                xb = (feat_fn(chunk, raw=True) - mu) / sd
                out.append(torch.sigmoid(model(torch.from_numpy(xb.astype(np.float32)).to(device))).squeeze(-1).cpu().numpy())
        return np.concatenate(out)
    best=-1; best_state=None
    for ep in range(epochs):
        model.train()
        for xb, yb in batches(tr, 4096, True):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = loss_fn(model(xb).squeeze(-1), yb); loss.backward(); opt.step()
        ap = average_precision_score(y_all[va], predict(va))
        if ap > best: best, best_state = ap, {k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict

def feat_abund(idx, raw=False):
    return np.log1p(X[idx].astype(np.float32))
def feat_seq(idx, raw=False):
    return np.asarray(E[np.sort(idx)] if False else E[idx], dtype=np.float32)

rows = []
for seed in SEEDS:
    ts = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    idx = np.arange(len(y_all))
    tr, te = train_test_split(idx, test_size=0.15, stratify=y_all, random_state=seed)
    tr, va = train_test_split(tr, test_size=0.15/0.85, stratify=y_all[tr], random_state=seed)
    yte = y_all[te]

    pred_ab = train_head(feat_abund, X.shape[1], tr, va)
    ab_te, ab_va = pred_ab(te), pred_ab(va)
    pred_sq = train_head(feat_seq, E.shape[1], tr, va)
    sq_te, sq_va = pred_sq(te), pred_sq(va)
    fuse = LogisticRegression(max_iter=1000).fit(np.c_[ab_va, sq_va], y_all[va])
    fz_te = fuse.predict_proba(np.c_[ab_te, sq_te])[:, 1]

    scores = {'MetaWIBELE unsup': mw_u_all[te], 'MetaWIBELE sup': mw_s_all[te],
              'Abundance MLP': ab_te, 'Sequence ESM-2 35M': sq_te, 'Fusion': fz_te}
    for name, s in scores.items():
        rows.append({'seed': seed, 'method': name,
                     'auprc': average_precision_score(yte, s),
                     'auroc': roc_auc_score(yte, s)})
    pd.DataFrame(rows).to_csv(PRED / '06_seed_results.csv', index=False)
    print(f'seed {seed} done in {time.time()-ts:.0f}s | '
          + ' '.join(f"{n}={average_precision_score(yte,s):.3f}" for n,s in scores.items()), flush=True)

df = pd.DataFrame(rows)
g = df.groupby('method')['auprc']
summary = pd.DataFrame({'mean': g.mean(), 'std': g.std(),
                        'ci95_lo': g.mean() - 1.96*g.std()/np.sqrt(len(SEEDS)),
                        'ci95_hi': g.mean() + 1.96*g.std()/np.sqrt(len(SEEDS))}).round(4)
summary.to_csv(PRED / '06_seed_summary.csv')
print(summary.to_string(), flush=True)
print('SEEDS_DONE', flush=True)
