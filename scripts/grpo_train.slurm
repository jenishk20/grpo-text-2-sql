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

# Fresh CUDA-12 conda env (torch 2.6+cu124 / vllm 0.8.5 / trl 0.17) — matches
# the 12.8 driver. It's a conda env, so activate with `source activate <path>`.
source activate /scratch/phalle.y/grpo_venv
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

# --no-vllm: vLLM 0.8.5 + TRL 0.17 misfire on platform detection (vllm_ascend
# import crash), so we use HF generation for rollouts — slower but reliable.
# Drop --no-vllm only once a TRL/vLLM pairing that imports cleanly is in place.
PYTHONUNBUFFERED=1 python -m src.train.grpo_train \
    --config configs/grpo_bird_7b.yaml \
    --no-vllm \
    --resume
