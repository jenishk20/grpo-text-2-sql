#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00          # match your queue's wall-clock limit
#SBATCH --job-name=bird-grpo-7b
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_grpo_7b_%j.out
#SBATCH --error=/scratch/phalle.y/bird_grpo_7b_%j.err
# GRPO training. Checkpoints every save_steps; --resume continues from the last
# checkpoint, so re-submit after a wall-clock timeout to keep going.

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/grpo-text-2-sql

# Keep ALL caches off $HOME (your home quota is full). XDG_CACHE_HOME is the
# catch-all so stray ~/.cache writes also land on scratch.
export HF_HOME=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
export XDG_CACHE_HOME=/scratch/phalle.y/.cache

# --- Weights & Biases (no ~/.netrc; auth via API key, config off $HOME) ---
# `wandb login` fails when HOME is full. Instead store your key once:
#   echo 'YOUR_WANDB_KEY' > /scratch/phalle.y/.wandb_key && chmod 600 /scratch/phalle.y/.wandb_key
export WANDB_API_KEY=$(cat /scratch/phalle.y/.wandb_key 2>/dev/null)
export WANDB_DIR=/scratch/phalle.y/wandb
export WANDB_CACHE_DIR=/scratch/phalle.y/wandb_cache
export WANDB_CONFIG_DIR=/scratch/phalle.y/wandb_config
# If compute nodes have no internet: log offline, then `wandb sync` from login.
# export WANDB_MODE=offline

# --no-vllm: this cluster's driver is CUDA 12.8, but vLLM 0.22 is a CUDA-13
# build that can't run here — so rollouts use HF generation (slower but works).
# Remove --no-vllm once a cu12 vLLM (<=0.18) is installed.
PYTHONUNBUFFERED=1 python -m src.train.grpo_train \
    --config configs/grpo_bird_7b.yaml \
    --no-vllm \
    --resume
