#!/usr/bin/env bash
#SBATCH --job-name=eval650
#SBATCH --partition=gpu
#SBATCH --account=uvicommsc_normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=eval650_%j.log

set -euo pipefail
cd "$HOME/taskb"
: "${DATA:=/data/uvicommsc/uvicommsc29}"
source /prod/precompiled/conda/etc/profile.d/conda.sh
conda activate "$DATA/.conda/envs/taskb"
python eval650.py
echo EVAL650_DONE
