#!/usr/bin/env python
"""High-dimensional cross-cohort transfer (companion to the species Task C).

Same train-on-HMP2 / test-on-Franzosa IBD-vs-control protocol, but on the
~2,000-dimensional Enzyme-Commission (EC) functional profile instead of 197
species. HMP2 community-level EC abundances come from HUMAnN (ecs_relab), the
Franzosa EC profile is Supplementary Dataset 6 (moesm8.xlsx). The two share
2,052 EC numbers. The question: does the deep representation's cross-cohort
advantage widen when the input is genuinely high-dimensional?

Models: ElasticNet, PCA(32)+LR, Autoencoder(32)+LR, VAE(32)+LR.
Outputs predictions/ec_results.csv and figures/fig17_ec_crosscohort.png.
"""
from pathlib import Path
import gzip, re, numpy as np, pandas as pd, warnings, torch, torch.nn as nn
warnings.simplefilter('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).parent; TA = HERE.parent / 'task_a'
PRED = HERE / 'predictions'; FIG = HERE / 'figures'; PRED.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eckey = lambda s: (re.match(r'^([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)', s) or [None, None])[1] if re.match(r'^([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)', s) else None

# ---------- HMP2 EC (community-level rows) ----------
data = {}
with gzip.open(TA / 'ecs_relab.tsv.gz', 'rt') as f:
    samples = f.readline().rstrip('\n').split('\t')[1:]
    for line in f:
        name = line.split('\t', 1)[0]
        if '|' in name or name.startswith(('UNMAPPED', 'UNGROUPED')):
            continue
        k = eckey(name)
        if k:
            data[k] = np.array(line.rstrip('\n').split('\t')[1:], dtype=np.float64)
hmp = pd.DataFrame(data, index=[s for s in samples])  # samples x EC
lab = pd.read_parquet(TA / 'processed' / 'labels.parquet').copy()
lab['base'] = lab['sample_id'].str.replace('_P$', '', regex=True)
lab = lab.drop_duplicates('base').set_index('base')
common = [s for s in hmp.index if s in lab.index]
hmp = hmp.loc[common]
y_h = lab.loc[common, 'diagnosis'].isin(['CD', 'UC']).astype(int).values
print(f'HMP2 EC matrix {hmp.shape} | IBD rate {y_h.mean():.3f}', flush=True)

# ---------- Franzosa EC (Supplementary Dataset 6) ----------
fr_raw = pd.read_excel(HERE / 'raw' / 'moesm8.xlsx', sheet_name='dataset_s6', header=1)
fr_raw = fr_raw.rename(columns={fr_raw.columns[0]: 'feature'})
META = ['SRA_metagenome_name', 'Age', 'Diagnosis', 'Fecal.Calprotectin', 'antibiotic', 'immunosuppressant', 'mesalamine', 'steroids']
y_f = fr_raw[fr_raw['feature'] == 'Diagnosis'].iloc[0, 1:].isin(['CD', 'UC']).astype(int).values
fr_ec = fr_raw[~fr_raw['feature'].isin(META)].copy()
fr_ec['ec'] = fr_ec['feature'].astype(str).map(eckey)
fr_ec = fr_ec.dropna(subset=['ec']).groupby('ec').first().drop(columns='feature')
fr = fr_ec.apply(pd.to_numeric, errors='coerce').T.fillna(0.0)   # samples x EC
print(f'Franzosa EC matrix {fr.shape} | IBD rate {y_f.mean():.3f}', flush=True)

shared = sorted(set(hmp.columns) & set(fr.columns))
print(f'shared EC numbers: {len(shared)}', flush=True)

def prep(df):
    X = df.reindex(columns=shared).fillna(0.0).values.astype(np.float64)
    X = X / np.clip(X.sum(1, keepdims=True), 1e-9, None)
    return np.log10(X + 1e-6)
Xh, Xf = prep(hmp), prep(fr)

# ---------- models ----------
class AE(nn.Module):
    def __init__(s, d, k=32):
        super().__init__(); s.enc = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Linear(256,64), nn.ReLU(), nn.Linear(64,k))
        s.dec = nn.Sequential(nn.Linear(k,64), nn.ReLU(), nn.Linear(64,256), nn.ReLU(), nn.Linear(256,d))
    def forward(s, x): z = s.enc(x); return s.dec(z), z
class VAE(nn.Module):
    def __init__(s, d, k=32):
        super().__init__(); s.enc = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Linear(256,64), nn.ReLU())
        s.mu = nn.Linear(64,k); s.lv = nn.Linear(64,k)
        s.dec = nn.Sequential(nn.Linear(k,64), nn.ReLU(), nn.Linear(64,256), nn.ReLU(), nn.Linear(256,d))
    def forward(s, x):
        h = s.enc(x); mu, lv = s.mu(h), s.lv(h); z = mu + torch.randn_like(mu)*torch.exp(0.5*lv); return s.dec(z), mu, lv

def fit_predict(kind, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    lr = lambda: LogisticRegression(max_iter=3000, class_weight='balanced')
    if kind == 'elasticnet':
        return LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=1.0, solver='saga', max_iter=6000, class_weight='balanced').fit(Xtr_s, ytr).predict_proba(Xte_s)[:, 1]
    if kind == 'pca':
        p = PCA(32, random_state=0).fit(Xtr_s); return lr().fit(p.transform(Xtr_s), ytr).predict_proba(p.transform(Xte_s))[:, 1]
    if kind == 'ae':
        torch.manual_seed(0); m = AE(Xtr_s.shape[1]).to(device); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-5)
        xb = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
        for _ in range(250):
            opt.zero_grad(); xr, _ = m(xb); nn.functional.mse_loss(xr, xb).backward(); opt.step()
        enc = lambda Z: m.enc(torch.tensor(Z, dtype=torch.float32, device=device)).detach().cpu().numpy()
        return lr().fit(enc(Xtr_s), ytr).predict_proba(enc(Xte_s))[:, 1]
    if kind == 'vae':
        torch.manual_seed(0); m = VAE(Xtr_s.shape[1]).to(device); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-5)
        xb = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
        for _ in range(250):
            opt.zero_grad(); xr, mu, lv = m(xb)
            (nn.functional.mse_loss(xr, xb) + 0.1*(-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp()))).backward(); opt.step()
        enc = lambda Z: (lambda h: m.mu(h))(m.enc(torch.tensor(Z, dtype=torch.float32, device=device))).detach().cpu().numpy()
        return lr().fit(enc(Xtr_s), ytr).predict_proba(enc(Xte_s))[:, 1]

MODELS = {'ElasticNet': 'elasticnet', 'PCA32 + LR': 'pca', 'Autoencoder32 + LR': 'ae', 'VAE32 + LR': 'vae'}
skf = StratifiedKFold(5, shuffle=True, random_state=0)
rows = []
for name, kind in MODELS.items():
    fr_cv = np.mean([roc_auc_score(y_f[te], fit_predict(kind, Xf[tr], y_f[tr], Xf[te])) for tr, te in skf.split(Xf, y_f)])
    p_h2f = fit_predict(kind, Xh, y_h, Xf); h2f = roc_auc_score(y_f, p_h2f); h2f_ap = average_precision_score(y_f, p_h2f)
    p_f2h = fit_predict(kind, Xf, y_f, Xh); f2h = roc_auc_score(y_h, p_f2h)
    rows.append({'model': name, 'Franzosa_CV_AUROC': round(fr_cv, 3), 'HMP2_to_Franzosa_AUROC': round(h2f, 3),
                 'HMP2_to_Franzosa_AUPRC': round(h2f_ap, 3), 'Franzosa_to_HMP2_AUROC': round(f2h, 3),
                 'mean_transfer_AUROC': round((h2f + f2h) / 2, 3)})
    print(f'{name}: Fr-CV {fr_cv:.3f} | HMP2->Fr {h2f:.3f} (AUPRC {h2f_ap:.3f}) | Fr->HMP2 {f2h:.3f}', flush=True)
res = pd.DataFrame(rows); res.to_csv(PRED / 'ec_results.csv', index=False)
print('\n', res.to_string(index=False), flush=True)

# ---------- fig17 ----------
x = np.arange(len(res)); w = 0.38
fig, ax = plt.subplots(figsize=(8.2, 4.7))
ax.bar(x - w/2, res['HMP2_to_Franzosa_AUROC'], w, label='HMP2 -> Franzosa', color='#c44e52', edgecolor='black', lw=0.6)
ax.bar(x + w/2, res['Franzosa_to_HMP2_AUROC'], w, label='Franzosa -> HMP2', color='#dd8452', edgecolor='black', lw=0.6)
ax.plot(x, res['Franzosa_CV_AUROC'], 'D', color='#2e7d32', ms=8, label='within-cohort CV (clean)')
for i, r in res.iterrows():
    ax.text(i - w/2, r['HMP2_to_Franzosa_AUROC'] + 0.01, f"{r['HMP2_to_Franzosa_AUROC']:.3f}", ha='center', fontsize=8)
    ax.text(i + w/2, r['Franzosa_to_HMP2_AUROC'] + 0.01, f"{r['Franzosa_to_HMP2_AUROC']:.3f}", ha='center', fontsize=8)
ax.axhline(0.5, ls=':', c='gray'); ax.set_xticks(x); ax.set_xticklabels(res['model']); ax.set_ylim(0, 1.0)
ax.set_ylabel('AUROC (IBD vs control)')
ax.set_title(f'High-dimensional companion: {len(shared)} shared EC enzymes')
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig(FIG / 'fig17_ec_crosscohort.png', dpi=150); plt.close()
print('wrote figures/fig17_ec_crosscohort.png', flush=True)
