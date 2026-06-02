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

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

# --- Weights & Biases ---
# Preferred: run `wandb login` once on the login node (writes ~/.netrc).
# Otherwise export your key here:
# export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# If the compute nodes have no internet, log offline and `wandb sync` later:
# export WANDB_MODE=offline
export WANDB_DIR=/scratch/phalle.y/wandb
export WANDB_CACHE_DIR=/scratch/phalle.y/wandb_cache

PYTHONUNBUFFERED=1 python -m src.train.grpo_train \
    --config configs/grpo_bird_7b.yaml \
    --resume
