#!/usr/bin/env bash
#SBATCH --job-name=esm650
#SBATCH --partition=gpu
#SBATCH --account=uvicommsc_normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=esm650_%j.log

# ESM-2 650M embeddings for Task B on CSUC (H100). Submit: cd ~/taskb && sbatch run_esm_650.sh
set -euo pipefail
cd "$HOME/taskb"
: "${DATA:=/data/uvicommsc/uvicommsc29}"

export http_proxy=http://192.168.255.254:8080
export https_proxy=http://192.168.255.254:8080
source /prod/precompiled/conda/etc/profile.d/conda.sh
conda activate "$DATA/.conda/envs/taskb"
export HF_HOME="$DATA/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python esm_embed.py --model facebook/esm2_t33_650M_UR50D --batch 32 --out processed/seq_emb_650M.npy
echo DONE_650
