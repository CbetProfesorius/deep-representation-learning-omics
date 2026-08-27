#!/usr/bin/env python
"""Task C - cross-cohort generalization (the central test of the thesis claim).

Train an IBD (CD/UC) vs non-IBD-control classifier on the HMP2/IBDMDB cohort
(MetaPhlAn species relative abundances) and evaluate it, with no refitting, on
an independent cohort: the PRISM + validation cohort of Franzosa et al. 2019
(Nat. Microbiol., Supplementary Dataset 4). The two cohorts are profiled with
the same MetaPhlAn2 vocabulary, so we restrict to the 197 species they share.

Three model classes are compared, to ask whether a learned (deep) representation
transfers across cohorts better than the classical linear pipelines:
  (1) ElasticNet logistic regression   - classical penalized linear baseline,
  (2) PCA(32) + logistic regression    - classical linear dimensionality reduction,
  (3) Autoencoder(32) + logistic reg.  - the deep unsupervised representation.

For each model we report in-cohort performance (participant-grouped 5-fold CV on
HMP2) and cross-cohort performance (train on all HMP2, test on Franzosa), and the
drop between them. Outputs predictions/taskc_results.csv and figures fig13/fig14.
"""
from pathlib import Path
import numpy as np, pandas as pd, warnings, torch, torch.nn as nn
warnings.simplefilter('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).parent
TA = HERE.parent / 'task_a'
PRED = HERE / 'predictions'; FIG = HERE / 'figures'
PRED.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(0); torch.manual_seed(0); np.random.seed(0)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# ---------- load HMP2 ----------
hmp = pd.read_parquet(TA / 'processed' / 'X_tax_species.parquet')
lab = pd.read_parquet(TA / 'processed' / 'labels.parquet').set_index('sample_id').loc[hmp.index]
y_hmp = lab['diagnosis'].isin(['CD', 'UC']).astype(int).values
groups = lab['participant_id'].values

# ---------- load Franzosa (Supplementary Dataset 4) ----------
fr_raw = pd.read_excel(HERE / 'raw' / 'moesm6.xlsx', sheet_name='dataset_s4', header=1)
fr_raw = fr_raw.rename(columns={fr_raw.columns[0]: 'feature'})
META = ['SRA_metagenome_name', 'Age', 'Diagnosis', 'Fecal.Calprotectin',
        'antibiotic', 'immunosuppressant', 'mesalamine', 'steroids']
diag = fr_raw[fr_raw['feature'] == 'Diagnosis'].iloc[0, 1:]
y_fr = diag.isin(['CD', 'UC']).astype(int).values
fr = fr_raw[~fr_raw['feature'].isin(META)].set_index('feature').apply(pd.to_numeric, errors='coerce').T.fillna(0.0)

# ---------- shared feature space ----------
shared = sorted(set(hmp.columns) & set(fr.columns))
print(f'shared species: {len(shared)} | HMP2 n={len(hmp)} IBD={y_hmp.mean():.3f} | '
      f'Franzosa n={len(fr)} IBD={y_fr.mean():.3f}', flush=True)

def prep(df):
    X = df.reindex(columns=shared).fillna(0.0).values.astype(np.float64)
    X = X / np.clip(X.sum(1, keepdims=True), 1e-9, None)   # re-close on shared species
    return np.log10(X + 1e-5)                              # log relative abundance
Xh, Xf = prep(hmp), prep(fr)

# ---------- model builders (each fits scaler on its own train fold) ----------
class AE(nn.Module):
    def __init__(self, d, k=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Linear(256,64), nn.ReLU(), nn.Linear(64,k))
        self.dec = nn.Sequential(nn.Linear(k,64), nn.ReLU(), nn.Linear(64,256), nn.ReLU(), nn.Linear(256,d))
    def forward(self, x): z = self.enc(x); return self.dec(z), z

def train_ae(Xtr, epochs=200, k=32):
    torch.manual_seed(0)
    m = AE(Xtr.shape[1], k).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    xb = torch.tensor(Xtr, dtype=torch.float32, device=device)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        xr, _ = m(xb); loss = nn.functional.mse_loss(xr, xb)
        loss.backward(); opt.step()
    m.eval(); return m

def encode(m, X):
    with torch.no_grad():
        _, z = m(torch.tensor(X, dtype=torch.float32, device=device))
    return z.cpu().numpy()

def fit_predict(kind, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    if kind == 'elasticnet':
        clf = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=1.0,
                                 solver='saga', max_iter=5000, class_weight='balanced')
        clf.fit(Xtr_s, ytr); return clf.predict_proba(Xte_s)[:, 1]
    if kind == 'pca':
        p = PCA(n_components=32, random_state=0).fit(Xtr_s)
        clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(p.transform(Xtr_s), ytr)
        return clf.predict_proba(p.transform(Xte_s))[:, 1]
    if kind == 'ae':
        m = train_ae(Xtr_s)
        clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(encode(m, Xtr_s), ytr)
        return clf.predict_proba(encode(m, Xte_s))[:, 1]

MODELS = {'ElasticNet': 'elasticnet', 'PCA32 + LR': 'pca', 'Autoencoder32 + LR': 'ae'}

def cv_auroc(kind, X, y, splitter, g=None):
    aus = []
    for tr, te in (splitter.split(X, y, g) if g is not None else splitter.split(X, y)):
        aus.append(roc_auc_score(y[te], fit_predict(kind, X[tr], y[tr], X[te])))
    return float(np.mean(aus)), float(np.std(aus))

def boot_ci(y, s, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); out = np.empty(B)
    for i in range(B):
        ix = rng.integers(0, n, n)
        yi = y[ix]
        out[i] = roc_auc_score(yi, s[ix]) if 0 < yi.sum() < n else np.nan
    return round(float(np.nanpercentile(out, 2.5)), 3), round(float(np.nanpercentile(out, 97.5)), 3)

# ---------- (a) the leakage trap: naive vs participant-grouped CV on HMP2 ----------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
naive_m, naive_s = cv_auroc('elasticnet', Xh, y_hmp, skf)
grp_m, grp_s = cv_auroc('elasticnet', Xh, y_hmp, sgkf, groups)
leak = pd.DataFrame([
    {'evaluation': 'HMP2 naive sample-level 5-fold CV (LEAKY)', 'AUROC': round(naive_m, 3), 'sd': round(naive_s, 3)},
    {'evaluation': 'HMP2 participant-grouped 5-fold CV (correct)', 'AUROC': round(grp_m, 3), 'sd': round(grp_s, 3)},
])
leak.to_csv(PRED / 'taskc_leakage.csv', index=False)
print('\n[leakage trap, ElasticNet on HMP2 species]\n', leak.to_string(index=False), flush=True)

# ---------- (b) bidirectional cross-cohort transfer + clean within-cohort reference ----------
rows = []; h2f_curves = {}
for name, kind in MODELS.items():
    # clean within-cohort reference: Franzosa is cross-sectional, so plain stratified CV is leakage-free
    fr_cv_m, fr_cv_s = cv_auroc(kind, Xf, y_fr, skf)
    # HMP2 -> Franzosa
    p_h2f = fit_predict(kind, Xh, y_hmp, Xf)
    h2f_au, h2f_ap, h2f_f1 = roc_auc_score(y_fr, p_h2f), average_precision_score(y_fr, p_h2f), f1_score(y_fr, (p_h2f >= .5).astype(int))
    h2f_lo, h2f_hi = boot_ci(y_fr, p_h2f)
    h2f_curves[name] = roc_curve(y_fr, p_h2f)
    # Franzosa -> HMP2
    p_f2h = fit_predict(kind, Xf, y_fr, Xh)
    f2h_au, f2h_ap, f2h_f1 = roc_auc_score(y_hmp, p_f2h), average_precision_score(y_hmp, p_f2h), f1_score(y_hmp, (p_f2h >= .5).astype(int))
    rows.append({'model': name,
                 'Franzosa_CV_AUROC': round(fr_cv_m, 3),
                 'HMP2_to_Franzosa_AUROC': round(h2f_au, 3),
                 'HMP2_to_Franzosa_CI': f'[{h2f_lo}, {h2f_hi}]',
                 'HMP2_to_Franzosa_AUPRC': round(h2f_ap, 3),
                 'Franzosa_to_HMP2_AUROC': round(f2h_au, 3), 'Franzosa_to_HMP2_AUPRC': round(f2h_ap, 3),
                 'mean_transfer_AUROC': round((h2f_au + f2h_au) / 2, 3)})
    rows[-1]['_h2f_lo'], rows[-1]['_h2f_hi'] = h2f_lo, h2f_hi
    print(f'{name}: Franzosa-CV {fr_cv_m:.3f} | HMP2->Fr {h2f_au:.3f} [{h2f_lo}, {h2f_hi}] | Fr->HMP2 {f2h_au:.3f}', flush=True)

res = pd.DataFrame(rows)
h2f_lo = res.pop('_h2f_lo').values; h2f_hi = res.pop('_h2f_hi').values
res.to_csv(PRED / 'taskc_results.csv', index=False)
print('\n', res.to_string(index=False), flush=True)

# ---------- fig13: bidirectional cross-cohort AUROC ----------
x = np.arange(len(res)); w = 0.38
fig, ax = plt.subplots(figsize=(8.0, 4.7))
h2f_err = np.c_[res['HMP2_to_Franzosa_AUROC'].values - h2f_lo, h2f_hi - res['HMP2_to_Franzosa_AUROC'].values].T
ax.bar(x - w/2, res['HMP2_to_Franzosa_AUROC'], w, yerr=h2f_err, capsize=4, label='HMP2 -> Franzosa (95% CI)', color='#c44e52', edgecolor='black', lw=0.6)
ax.bar(x + w/2, res['Franzosa_to_HMP2_AUROC'], w, label='Franzosa -> HMP2', color='#dd8452', edgecolor='black', lw=0.6)
ax.plot(x, res['Franzosa_CV_AUROC'], 'D', color='#2e7d32', ms=8, label='within-cohort CV (clean reference)')
for i, r in res.iterrows():
    ax.text(i - w/2, r['HMP2_to_Franzosa_AUROC'] + 0.01, f"{r['HMP2_to_Franzosa_AUROC']:.3f}", ha='center', fontsize=8)
    ax.text(i + w/2, r['Franzosa_to_HMP2_AUROC'] + 0.01, f"{r['Franzosa_to_HMP2_AUROC']:.3f}", ha='center', fontsize=8)
ax.axhline(0.5, ls=':', c='gray', lw=1)
ax.set_xticks(x); ax.set_xticklabels(res['model']); ax.set_ylim(0, 1.0)
ax.set_ylabel('AUROC (IBD vs control)')
ax.set_title('Cross-cohort transfer (197 shared species)')
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig(FIG / 'fig13_crosscohort_auroc.png', dpi=150); plt.close()

# ---------- fig14: ROC curves HMP2 -> Franzosa ----------
fig, ax = plt.subplots(figsize=(5.6, 5.2))
for (name, (fpr, tpr, _)), c in zip(h2f_curves.items(), ['#4c72b0', '#dd8452', '#2e7d32']):
    ax.plot(fpr, tpr, label=f"{name} (AUROC {res.loc[res.model==name,'HMP2_to_Franzosa_AUROC'].values[0]:.3f})", color=c, lw=1.6)
ax.plot([0,1],[0,1], ls=':', c='gray')
ax.set_xlabel('false-positive rate'); ax.set_ylabel('true-positive rate')
ax.set_title('Cross-cohort ROC: HMP2-trained models on the Franzosa cohort')
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig(FIG / 'fig14_crosscohort_roc.png', dpi=150); plt.close()
print('wrote figures/fig13_crosscohort_auroc.png, fig14_crosscohort_roc.png', flush=True)
