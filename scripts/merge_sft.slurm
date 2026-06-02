#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1        # merge runs on CPU; any node is fine, GPU not required
#SBATCH --time=00:30:00
#SBATCH --job-name=bird-merge-sft
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/bird_merge_sft_%j.out
#SBATCH --error=/scratch/phalle.y/bird_merge_sft_%j.err
# One-time: fold the BIRD SFT LoRA into the base -> a full SFT-merged model.

source activate /scratch/phalle.y/py310env
cd /scratch/phalle.y/grpo-text-2-sql

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache
export TORCH_HOME=/scratch/phalle.y/torch_cache
export TRITON_CACHE_DIR=/scratch/phalle.y/triton_cache
export PIP_CACHE_DIR=/scratch/phalle.y/pip_cache

PYTHONUNBUFFERED=1 python -m src.train.merge_adapter \
    --base    Qwen/Qwen2.5-Coder-7B-Instruct \
    --adapter /scratch/phalle.y/bird_sft_adapter_7b/final_adapter \
    --out     /scratch/phalle.y/sft_merged_7b
