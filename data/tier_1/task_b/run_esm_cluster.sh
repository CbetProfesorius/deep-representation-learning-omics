#!/usr/bin/env bash
#SBATCH --job-name=esm_embed
#SBATCH --partition=gpu
#SBATCH --account=uvicommsc_normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=esm_embed_%j.log
# NOTE: the gpu partition uses per-GPU memory/CPU defaults (auto ~32 CPUs/GPU);
# specifying --mem or --cpus-per-task here causes "node configuration not available".

# Precompute ESM-2 embeddings for Task B on CSUC (pirineus3, H100 + Slurm).
# Submit with:  cd ~/taskb && sbatch run_esm_cluster.sh

set -euo pipefail
cd "$HOME/taskb"

# $DATA may not be exported into the batch environment; provide a fallback.
: "${DATA:=/data/uvicommsc/uvicommsc29}"

# Proxy (compute nodes reach the internet only through this); harmless if offline cache is used.
export http_proxy=http://192.168.255.254:8080
export https_proxy=http://192.168.255.254:8080

# Personal conda env (work_env is read-only and lacks torch/transformers).
source /prod/precompiled/conda/etc/profile.d/conda.sh
conda activate "$DATA/.conda/envs/taskb"

# Use prefetched model weights so we don't depend on compute-node networking.
export HF_HOME="$DATA/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -c "import transformers, torch; print('transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

# First pass: 35M model -> 480-dim embeddings (~2.8 GB), fast, validates the path.
python esm_embed.py --model facebook/esm2_t12_35M_UR50D --batch 128 --out processed/seq_emb_35M.npy

# Final number: 650M -> 1280-dim (~7.4 GB). Uncomment after the 35M pass succeeds.
# python esm_embed.py --model facebook/esm2_t33_650M_UR50D --batch 32 --out processed/seq_emb_650M.npy

echo "embeddings written to processed/"
