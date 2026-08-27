#!/usr/bin/env python
"""Robustness to noise / missing inputs (Objective 3) and extra deep
representations (Objective 1) on Task A.

Setup: HMP2 community-level MetaCyc pathways (466 features), dysbiosis label,
the same participant-level train/val/test split as the main Task A run. Models:
  - ElasticNet                       (classical linear)
  - PCA(32) + LR                     (classical linear DR)
  - Autoencoder(32) + LR             (deep, plain reconstruction)
  - Denoising AE(32) + LR            (deep, trained on noised inputs)   [new]
  - VAE(32) + LR                     (deep, variational)                [new]

We measure (a) clean test AUROC and (b) how AUROC degrades when the test inputs
are corrupted with additive Gaussian noise or random feature missingness
(imputed to the training mean), averaged over repeated random corruptions.
The question is whether the learned encoders degrade more gracefully than the
linear pipelines. Outputs predictions/robust_results.csv, robust_clean.csv and
figures fig15 (degradation curves) + fig16 (clean accuracy).
"""
from pathlib import Path
import json, numpy as np, pandas as pd, warnings, torch, torch.nn as nn
warnings.simplefilter('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).parent; PROC = HERE / 'processed'
PRED = HERE / 'predictions'; FIG = HERE / 'figures'
PRED.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

X = pd.read_parquet(PROC / 'X_pathway_community.parquet')
lab = pd.read_parquet(PROC / 'labels.parquet').set_index('sample_id').loc[X.index]
y = lab['is_dysbiotic'].astype(int).values
split = json.load(open(PROC / 'split.json'))['split']
part = lab['participant_id'].values
mask = np.array([split.get(p, 'train') for p in part])
tr, va, te = (mask == 'train'), (mask == 'val'), (mask == 'test')
Xlog = np.log10(X.values.astype(np.float64) + 1e-6)
sc = StandardScaler().fit(Xlog[tr])
Xs = sc.transform(Xlog)
D = Xs.shape[1]
print(f'pathways D={D} | train {tr.sum()} val {va.sum()} test {te.sum()} | pos rate {y[tr].mean():.3f}', flush=True)

# ---------------- encoders ----------------
class AE(nn.Module):
    def __init__(s, d, k=32):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Linear(256,64), nn.ReLU(), nn.Linear(64,k))
        s.dec = nn.Sequential(nn.Linear(k,64), nn.ReLU(), nn.Linear(64,256), nn.ReLU(), nn.Linear(256,d))
    def forward(s, x): z = s.enc(x); return s.dec(z), z

class VAE(nn.Module):
    def __init__(s, d, k=32):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Linear(256,64), nn.ReLU())
        s.mu = nn.Linear(64, k); s.lv = nn.Linear(64, k)
        s.dec = nn.Sequential(nn.Linear(k,64), nn.ReLU(), nn.Linear(64,256), nn.ReLU(), nn.Linear(256,d))
    def forward(s, x):
        h = s.enc(x); mu, lv = s.mu(h), s.lv(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5*lv)
        return s.dec(z), mu, lv

def train_ae(Xtr, denoise=0.0, epochs=300, k=32):
    torch.manual_seed(0); m = AE(Xtr.shape[1], k).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    xb = torch.tensor(Xtr, dtype=torch.float32, device=device)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        inp = xb + denoise*torch.randn_like(xb) if denoise else xb
        xr, _ = m(inp); loss = nn.functional.mse_loss(xr, xb)
        loss.backward(); opt.step()
    m.eval(); return lambda Z: encode_ae(m, Z)

def encode_ae(m, Z):
    with torch.no_grad():
        _, z = m(torch.tensor(Z, dtype=torch.float32, device=device))
    return z.cpu().numpy()

def train_vae(Xtr, epochs=300, k=32, beta=0.1):
    torch.manual_seed(0); m = VAE(Xtr.shape[1], k).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    xb = torch.tensor(Xtr, dtype=torch.float32, device=device)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        xr, mu, lv = m(xb)
        rec = nn.functional.mse_loss(xr, xb)
        kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        (rec + beta*kl).backward(); opt.step()
    m.eval()
    def enc(Z):
        with torch.no_grad():
            h = m.enc(torch.tensor(Z, dtype=torch.float32, device=device)); return m.mu(h).cpu().numpy()
    return enc

# ---------------- build the 5 fitted models (transform + classifier) ----------------
def lr(): return LogisticRegression(max_iter=3000, class_weight='balanced')
models = {}
# ElasticNet
en = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=1.0, solver='saga', max_iter=8000, class_weight='balanced').fit(Xs[tr], y[tr])
models['ElasticNet'] = (lambda Z: Z, en)
# PCA
pca = PCA(n_components=32, random_state=0).fit(Xs[tr])
models['PCA32 + LR'] = (pca.transform, lr().fit(pca.transform(Xs[tr]), y[tr]))
# AE / DAE / VAE
enc_ae = train_ae(Xs[tr], denoise=0.0)
models['Autoencoder32 + LR'] = (enc_ae, lr().fit(enc_ae(Xs[tr]), y[tr]))
enc_dae = train_ae(Xs[tr], denoise=1.0)
models['Denoising AE32 + LR'] = (enc_dae, lr().fit(enc_dae(Xs[tr]), y[tr]))
enc_vae = train_vae(Xs[tr])
models['VAE32 + LR'] = (enc_vae, lr().fit(enc_vae(Xs[tr]), y[tr]))

def auroc_for(transform, clf, Xeval):
    p = clf.predict_proba(transform(Xeval))[:, 1]
    return roc_auc_score(y[te], p)

clean = {name: round(auroc_for(t, c, Xs[te]), 3) for name, (t, c) in models.items()}
pd.DataFrame([{'model': k, 'clean_test_AUROC': v} for k, v in clean.items()]).to_csv(PRED / 'robust_clean.csv', index=False)
print('clean test AUROC:', clean, flush=True)

# ---------------- corruption sweeps ----------------
Xte = Xs[te].copy(); n_rep = 20
noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
miss_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
rows = []
for name, (t, c) in models.items():
    for sig in noise_levels:
        aus = []
        for r in range(n_rep):
            rng = np.random.default_rng(r)
            Xc = Xte + (sig * rng.standard_normal(Xte.shape) if sig else 0.0)
            aus.append(roc_auc_score(y[te], c.predict_proba(t(Xc))[:, 1]))
        rows.append({'model': name, 'corruption': 'gaussian_noise', 'level': sig,
                     'auroc': round(np.mean(aus), 4), 'sd': round(np.std(aus), 4)})
    for p in miss_levels:
        aus = []
        for r in range(n_rep):
            rng = np.random.default_rng(100 + r)
            Xc = Xte.copy()
            if p:
                m = rng.random(Xc.shape) < p
                Xc[m] = 0.0   # standardized -> 0 == impute to training mean
            aus.append(roc_auc_score(y[te], c.predict_proba(t(Xc))[:, 1]))
        rows.append({'model': name, 'corruption': 'missingness', 'level': p,
                     'auroc': round(np.mean(aus), 4), 'sd': round(np.std(aus), 4)})
res = pd.DataFrame(rows)
res.to_csv(PRED / 'robust_results.csv', index=False)

# retention (AUROC relative to clean) for a compact summary
piv = res.pivot_table(index='model', columns=['corruption', 'level'], values='auroc')
print('\n', piv.round(3).to_string(), flush=True)

# ---------------- fig15: degradation curves ----------------
colors = {'ElasticNet': '#9e9e9e', 'PCA32 + LR': '#4c72b0', 'Autoencoder32 + LR': '#2e7d32',
          'Denoising AE32 + LR': '#c44e52', 'VAE32 + LR': '#8172b3'}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (corr, levels, xl) in zip(axes, [('gaussian_noise', noise_levels, 'added Gaussian noise σ (std units)'),
                                          ('missingness', miss_levels, 'fraction of features missing')]):
    for name in models:
        d = res[(res.model == name) & (res.corruption == corr)].sort_values('level')
        ax.plot(d['level'], d['auroc'], 'o-', label=name, color=colors[name], lw=1.6, ms=4)
    ax.axhline(0.5, ls=':', c='gray', lw=1); ax.set_xlabel(xl); ax.set_ylabel('test AUROC'); ax.set_ylim(0.45, 0.97)
axes[0].set_title('Robustness to input noise'); axes[1].set_title('Robustness to missing features')
axes[0].legend(fontsize=7.5, loc='lower left')
plt.tight_layout(); plt.savefig(FIG / 'fig15_robustness.png', dpi=150); plt.close()

# ---------------- fig16: clean accuracy ----------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
names = list(clean); vals = [clean[n] for n in names]
ax.bar(names, vals, color=[colors[n] for n in names], edgecolor='black', lw=0.6)
for i, v in enumerate(vals): ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
ax.set_ylim(0, 1.0); ax.set_ylabel('clean test AUROC (dysbiosis, pathways)')
ax.set_title('Clean accuracy of classical vs deep representations')
plt.setp(ax.get_xticklabels(), rotation=12, ha='right')
plt.tight_layout(); plt.savefig(FIG / 'fig16_clean_models.png', dpi=150); plt.close()
print('wrote fig15_robustness.png, fig16_clean_models.png', flush=True)
