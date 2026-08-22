#!/usr/bin/env python
"""Cluster-side Task B v1 eval for the ESM-2 650M embeddings.

Runs entirely on the cluster (the 7.4 GB embedding matrix never leaves it).
Reproduces the identical split as 02/03, trains the sequence-only MLP on the
650M embeddings, loads the locally-precomputed abundance scores
(processed/abund_scores_vate.parquet), fits the leakage-free val fusion, and
writes the small results to predictions/ for copy-back.
"""
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

torch.manual_seed(0); np.random.seed(0)
PROC = Path('processed'); PRED = Path('predictions'); PRED.mkdir(exist_ok=True)
EMB_FILE = PROC / 'seq_emb_650M.npy'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', device, 'emb', EMB_FILE, flush=True)

labels = pd.read_parquet(PROC / 'labels.parquet')
fids = pd.read_parquet(PROC / 'family_ids.parquet')['family_id'].values
assert (labels['family_id'].values == fids).all()
y_all = labels['is_bioactive'].astype(np.int8).values

idx = np.arange(len(y_all))
tr, te = train_test_split(idx, test_size=0.15, stratify=y_all, random_state=0)
tr, va = train_test_split(tr, test_size=0.15 / 0.85, stratify=y_all[tr], random_state=0)
for nm, ix in [('train', tr), ('val', va), ('test', te)]:
    print(f'  {nm:5s} {len(ix):>9,} pos={int(y_all[ix].sum()):>6,} rate={y_all[ix].mean():.4f}', flush=True)

full = np.load(EMB_FILE, mmap_mode='r')
EMB_DIM = full.shape[1]
print('emb shape', full.shape, flush=True)

# standardize on train
tr_mat = np.asarray(full[np.sort(tr)], dtype=np.float32)
mu = tr_mat.mean(0); sd = tr_mat.std(0) + 1e-6
del tr_mat

class EmbDS(Dataset):
    def __init__(self, ix): self.ix = np.sort(ix)
    def __len__(self): return len(self.ix)
    def __getitem__(self, i):
        g = int(self.ix[i])
        x = (np.asarray(full[g], dtype=np.float32) - mu) / sd
        return torch.from_numpy(x), torch.tensor(float(y_all[g]))

BATCH = 4096
tr_dl = DataLoader(EmbDS(tr), batch_size=BATCH, shuffle=True, num_workers=4)
va_ds, te_ds = EmbDS(va), EmbDS(te)
va_dl = DataLoader(va_ds, batch_size=BATCH, num_workers=4)
te_dl = DataLoader(te_ds, batch_size=BATCH, num_workers=4)
pos_weight = torch.tensor([(len(tr) - y_all[tr].sum()) / max(y_all[tr].sum(), 1)],
                          dtype=torch.float32, device=device)

class SeqMLP(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Dropout(0.3),
                              nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))
    def forward(s, x): return s.net(x)

def predict(model, dl):
    model.eval(); out = []
    with torch.no_grad():
        for xb, _ in dl:
            out.append(torch.sigmoid(model(xb.to(device))).squeeze(-1).cpu().numpy())
    return np.concatenate(out)

def train(model, epochs=10, lr=1e-3, wd=1e-5):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best, best_state = -1, None
    for ep in range(1, epochs+1):
        model.train(); ls=0; n=0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward(); opt.step(); ls += loss.item()*len(xb); n += len(xb)
        va_ap = average_precision_score(y_all[va_ds.ix], predict(model, va_dl))
        print(f'  ep {ep}/{epochs} loss={ls/n:.4f} val AUPRC={va_ap:.4f}', flush=True)
        if va_ap > best: best, best_state = va_ap, {k: v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); return model

print('training sequence-only ranker (ESM-2 650M)...', flush=True)
seq_model = train(SeqMLP(EMB_DIM), epochs=10)
seq_va = predict(seq_model, va_dl)   # aligned to va_ds.ix (sorted va)
seq_te = predict(seq_model, te_dl)   # aligned to te_ds.ix (sorted te)

# abundance scores -> align by family_id to the sorted val/test order
ab = pd.read_parquet(PROC / 'abund_scores_vate.parquet')
ab_map = dict(zip(ab['family_id'], ab['abund_score']))
abund_va = np.array([ab_map[f] for f in fids[va_ds.ix]], dtype=float)
abund_te = np.array([ab_map[f] for f in fids[te_ds.ix]], dtype=float)

fuse = LogisticRegression(max_iter=1000).fit(np.c_[abund_va, seq_va], y_all[va_ds.ix])
fusion_te = fuse.predict_proba(np.c_[abund_te, seq_te])[:, 1]

y_te = y_all[te_ds.ix]
def report(name, sc):
    sc = np.asarray(sc, float); ok = ~np.isnan(sc)
    out = {'method': name, 'auprc': average_precision_score(y_te[ok], sc[ok]),
           'auroc': roc_auc_score(y_te[ok], sc[ok])}
    for k in (100, 1000):
        kk = min(k, ok.sum()-1); top = np.argpartition(-sc[ok], kk)[:kk]
        out[f'P@{k}'] = y_te[ok][top].mean(); out[f'enr@{k}'] = out[f'P@{k}']/max(y_te[ok].mean(),1e-9)
    return out

mw_u = labels['metawibele_unsup_rank'].fillna(0).values[te_ds.ix]
mw_s = labels['metawibele_sup_rank'].fillna(0).values[te_ds.ix]
rows = [report('Sequence-only (ESM-2 650M)', seq_te),
        report('MetaWIBELE unsup', mw_u),
        report('MetaWIBELE sup', mw_s),
        report('Abundance v0 (MLP)', abund_te),
        report('Fusion (abundance + ESM-2 650M)', fusion_te)]
res = pd.DataFrame(rows).set_index('method').round(4)
print(res.to_string(), flush=True)
res.to_csv(PRED / '03_results_precomputed650.csv')
pd.DataFrame({'family_id': fids[te_ds.ix], 'y_true': y_te,
             'seq_score': seq_te, 'fusion_score': fusion_te}).to_parquet(
    PRED / '03_predictions_precomputed650.parquet', index=False)
print('wrote predictions/03_results_precomputed650.csv', flush=True)
