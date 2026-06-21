#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1        # eval is light; any GPU >=24GB works
#SBATCH --time=04:00:00          # ~3-4h for the full 1534-question dev set
#SBATCH --job-name=bird-grpo-eval
#SBATCH --mem=60GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_grpo_eval_%j.out
#SBATCH --error=/scratch/phalle.y/bird_grpo_eval_%j.err
# Evaluate the trained GRPO adapter on the BIRD dev set (execution accuracy).

source activate /scratch/phalle.y/grpo_venv
cd /scratch/phalle.y/grpo-text-2-sql

# Keep all caches off $HOME (home quota is full).
export HF_HOME=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache
export XDG_CACHE_HOME=/scratch/phalle.y/.cache
export WANDB_API_KEY=$(cat /scratch/phalle.y/.wandb_key 2>/dev/null)
export WANDB_DIR=/scratch/phalle.y/wandb
export WANDB_CACHE_DIR=/scratch/phalle.y/wandb_cache
export WANDB_CONFIG_DIR=/scratch/phalle.y/wandb_config
# export WANDB_MODE=offline      # if compute nodes have no internet

PYTHONUNBUFFERED=1 python -m src.eval.evaluate \
    --backend vllm \
    --model    /scratch/phalle.y/sft_merged_7b \
    --adapter  /scratch/phalle.y/results_bird_grpo_7b/final \
    --dev-json /scratch/phalle.y/bird_dev/dev_20240627/dev.json \
    --db-root  /scratch/phalle.y/bird_dev/dev_20240627/dev_databases \
    --out      /scratch/phalle.y/results_bird_grpo_7b/grpo_dev_eval.json \
    --wandb-project bird-grpo-7b \
    --wandb-run     grpo-bird-7b-dev
