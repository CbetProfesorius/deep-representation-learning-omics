#!/usr/bin/env python
"""Precompute ESM-2 embeddings for all Task B protein families.

Runs on the CSUC cluster GPU. Reads processed/sequences.parquet (family_id, sequence),
mean-pools the last hidden state of an ESM-2 model per sequence, and writes
processed/seq_emb.npy (float32, n_families x hidden_dim) in the SAME row order as
processed/family_ids.parquet (so it aligns with abundance.npy and labels.parquet).

Resumable: writes a memmap and a progress marker, so a re-run continues where it stopped.

Usage:
    python esm_embed.py --model facebook/esm2_t33_650M_UR50D --batch 16
    python esm_embed.py --model facebook/esm2_t12_35M_UR50D --batch 64   # smaller/faster

The 650M model -> hidden_dim 1280 -> ~7.4 GB for 1.45M families.
The 35M  model -> hidden_dim 480  -> ~2.8 GB.
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='facebook/esm2_t33_650M_UR50D')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max-len', type=int, default=1022)
    ap.add_argument('--proc', default='processed')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    proc = Path(args.proc)
    out_path = Path(args.out) if args.out else proc / 'seq_emb.npy'
    prog_path = out_path.with_suffix('.progress')

    fids = pd.read_parquet(proc / 'family_ids.parquet')['family_id'].values
    seqs = pd.read_parquet(proc / 'sequences.parquet')
    seq_map = dict(zip(seqs['family_id'], seqs['sequence']))
    seq_list = [seq_map.get(f, '') for f in fids]
    n = len(seq_list)

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'[esm_embed] device={device}  model={args.model}  n={n:,}  batch={args.batch}', flush=True)

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    if device == 'cuda':
        model = model.half()  # fp16 on GPU to fit 650M comfortably
    dim = model.config.hidden_size
    print(f'[esm_embed] hidden_dim={dim}  estimated size={n*dim*4/1e9:.1f} GB', flush=True)

    # resumable memmap
    emb = np.lib.format.open_memmap(out_path, mode='r+' if out_path.exists() else 'w+',
                                    dtype=np.float32, shape=(n, dim))
    start = int(prog_path.read_text()) if prog_path.exists() else 0
    print(f'[esm_embed] resuming at {start:,}', flush=True)

    t0 = time.time()
    with torch.no_grad():
        for s in range(start, n, args.batch):
            chunk = [seq[:args.max_len] if seq else 'A' for seq in seq_list[s:s + args.batch]]
            enc = tok(chunk, return_tensors='pt', padding=True, truncation=True, max_length=args.max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            hs = model(**enc).last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1).to(hs.dtype)
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
            emb[s:s + len(chunk)] = pooled.float().cpu().numpy()
            if s % (args.batch * 100) == 0:
                emb.flush(); prog_path.write_text(str(s))
                rate = (s - start + 1) / max(time.time() - t0, 1e-9)
                eta = (n - s) / max(rate, 1e-9) / 3600
                print(f'[esm_embed] {s:,}/{n:,}  {rate:.0f} seq/s  ETA {eta:.1f} h', flush=True)
    emb.flush(); prog_path.write_text(str(n))
    print(f'[esm_embed] done in {(time.time()-t0)/3600:.2f} h -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
