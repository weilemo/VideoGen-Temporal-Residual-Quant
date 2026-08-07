#!/usr/bin/env bash
#SBATCH -J qvg_sf_bf16
#SBATCH -p a100_local_A100-0311-17100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/slurm_logs/%x-%j.out
#SBATCH -e /data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/slurm_logs/%x-%j.err

set -euo pipefail

source /mnt/public/apps/miniconda3/etc/profile.d/conda.sh
conda activate self-forcing

cd /data2/moweile-20251213/workspace/videoquant/Quant-VideoGen
mkdir -p slurm_logs results/selfforcing/bf16

echo "hostname=$(hostname)"
echo "start_time=$(date '+%F %T')"
echo "python=$(which python)"
nvidia-smi

bash scripts/Self-Forcing/run_bf16.sh

echo "end_time=$(date '+%F %T')"
