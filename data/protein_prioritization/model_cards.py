#!/usr/bin/env python
"""Compute exact parameter counts for every network used in the thesis and emit
(1) a CSV/Markdown parameter table and (2) a block-diagram figure of the
abundance/sequence/fusion pipeline. Numbers are code-derived, not hand-counted.
"""
from pathlib import Path
import torch, torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path('figures'); FIG.mkdir(exist_ok=True)
PRED = Path('predictions')

def n_params(m):
    return sum(p.numel() for p in m.parameters())

def mlp(dims, dropout=None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout: layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

def autoencoder(d):
    enc = mlp([d, 256, 64, 32])
    dec = mlp([32, 64, 256, d])
    return nn.Sequential(enc, dec)

rows = []
def add(task, name, model, dims, note):
    rows.append({'task': task, 'model': name, 'architecture': dims,
                 'parameters': n_params(model), 'note': note})

# Task A
add('A', 'Autoencoder (pathways)', autoencoder(466), '466-256-64-32-64-256-466', 'unsupervised; +33-param LR head')
add('A', 'Autoencoder (species)',  autoencoder(572), '572-256-64-32-64-256-572', 'unsupervised; +33-param LR head')

# Task B v0
add('B v0', 'Linear (abundance)', mlp([1595, 1]),                 '1595-1',         'logistic baseline')
add('B v0', 'MLP (abundance)',    mlp([1595, 256, 64, 1], 0.3),   '1595-256-64-1',  'dropout 0.3')

# Task B v1 sequence MLPs
add('B v1', 'SeqMLP (k-mer)',     mlp([400, 256, 64, 1], 0.3),    '400-256-64-1',   'dropout 0.3')
add('B v1', 'SeqMLP (ESM-2 35M)', mlp([480, 256, 64, 1], 0.3),    '480-256-64-1',   'dropout 0.3')
add('B v1', 'SeqMLP (ESM-2 650M)',mlp([1280, 256, 64, 1], 0.3),   '1280-256-64-1',  'dropout 0.3')

# Fusion calibrator = logistic regression on 2 scores
rows.append({'task': 'B v1', 'model': 'Fusion (logistic)', 'architecture': '2-1',
             'parameters': 3, 'note': '2 weights + bias; fit on val'})

df = pd.DataFrame(rows)
df.to_csv(PRED / 'model_param_counts.csv', index=False)
print(df.to_string(index=False))

# pretrained backbones (reference, not trained here)
backbones = {'ESM-2 t12 (35M)': 33_500_000, 'ESM-2 t33 (650M)': 650_000_000}
print('\nfrozen backbones (reference):', backbones)

# ---------- block diagram ----------
fig, ax = plt.subplots(figsize=(9.5, 5.0)); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.08',
                 fc=fc, ec='black', lw=1.1))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14, lw=1.1, color='#333'))

# abundance branch
box(0.2, 4.2, 1.9, 0.9, 'Abundance\n(1595-dim)', '#cfe0f3')
box(2.5, 4.2, 2.2, 0.9, 'MLP\n1595-256-64-1', '#4c72b0')
arrow(2.1, 4.65, 2.5, 4.65)
box(5.1, 4.2, 1.6, 0.9, 'abundance\nscore', '#dbe9d8')
arrow(4.7, 4.65, 5.1, 4.65)
# sequence branch
box(0.2, 0.9, 1.9, 0.9, 'Protein seq\n(variable len)', '#f3e3cf')
box(2.5, 0.9, 2.2, 0.9, 'ESM-2 650M\n(frozen)\nmean-pool -> 1280', '#dd8452')
arrow(2.1, 1.35, 2.5, 1.35)
box(5.1, 0.9, 2.2, 0.9, 'SeqMLP\n1280-256-64-1', '#2e7d32')
arrow(4.7, 1.35, 5.1, 1.35)
box(7.6, 0.9, 1.6, 0.9, 'sequence\nscore', '#dbe9d8')
arrow(7.3, 1.35, 7.6, 1.35)
# fusion
box(7.9, 2.55, 1.8, 0.95, 'Fusion\nlogistic(2-1)\nfit on val', '#c44e52')
arrow(6.7, 4.65, 8.8, 3.5)   # abundance score -> fusion
arrow(8.4, 1.8, 8.7, 2.55)   # sequence score -> fusion
box(7.9, 4.5, 1.8, 0.8, 'bioactivity\nrank', '#eeeeee')
arrow(8.8, 3.5, 8.8, 4.5)
ax.set_title('Fusion architecture (abundance MLP + ESM-2 sequence MLP -> logistic fusion)', fontsize=10)
plt.tight_layout(); plt.savefig(FIG / 'fig8_architecture.png', dpi=150); plt.close()
print('wrote figures/fig8_architecture.png')
