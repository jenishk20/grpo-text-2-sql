# GRPO Text-to-SQL — BIRD · Qwen2.5-Coder-7B

A clean, reproducible **GRPO** (Group Relative Policy Optimization) pipeline that
RL-tunes `Qwen2.5-Coder-7B-Instruct` on the **BIRD** benchmark, starting from an
existing **BIRD SFT LoRA adapter** and rewarding SQL by *execution result*.

```
Qwen2.5-Coder-7B  +  bird_sft_adapter_7b
        │  (merge once)
        ▼
   sft_merged_7b ──► GRPO (fresh LoRA)
        ▲                  │  for each question:
        │                  │   • vLLM samples G candidate SQLs (a "group")
        └── reward ◄───────┘   • each SQL is executed on the SQLite DB
            (exec result          • reward = 1.0 if result set == gold, else
             vs gold SQL)           partial credit for valid/runnable SQL
                                  • GRPO pushes the policy toward the
                                    above-average samples in each group
```

Why GRPO here: no reward model, no preference pairs to build — the SQL executor
*is* the reward. This sidesteps the failure mode of the earlier DPO attempts
(gold-fallback pairs dominating and dragging the policy off the SFT base).

---

## Layout

```
grpo-text-2-sql/
├── configs/grpo_bird_7b.yaml      # all hyperparameters + cluster paths
├── src/
│   ├── shared/                    # reused by training AND eval
│   │   ├── prompt.py              # the one prompt template (must match SFT)
│   │   ├── schema_loader.py       # SQLite schema → DDL string (cached)
│   │   ├── sqlite_executor.py     # read-only exec with hard timeout
│   │   ├── evaluator.py           # order-insensitive result-set match
│   │   └── sql_utils.py           # extract_sql() from model output
│   ├── data/build_dataset.py      # BIRD json → HF Dataset (drops over-long prompts)
│   ├── rewards/sql_reward.py      # execution-based GRPO reward (factory)
│   ├── train/
│   │   ├── merge_adapter.py       # SFT LoRA → merged base (run once)
│   │   └── grpo_train.py          # TRL GRPOTrainer + vLLM colocate
│   └── eval/evaluate.py           # BIRD dev: accuracy + difficulty breakdown
├── scripts/
│   ├── check_setup.py             # CPU pre-flight: paths + reward sanity
│   ├── merge_sft.slurm            # SBATCH: conda + caches + merge
│   ├── grpo_train.slurm           # SBATCH: GRPO training (+ wandb)
│   └── evaluate.slurm             # SBATCH: BIRD dev eval (+ wandb)
└── requirements.txt
```

Every script runs as a module from the repo root: `python -m src.<...>`.

---

## Setup on the cluster

```bash
cd /scratch/phalle.y
git clone <your-remote>/grpo-text-2-sql.git
cd grpo-text-2-sql

# Use your Python 3.10+ conda env. Install torch matching the cluster CUDA
# first, then the rest:
source activate /scratch/phalle.y/py310env
pip install -r requirements.txt

# W&B auth WITHOUT writing to $HOME (home quota is full -> `wandb login` fails):
echo 'YOUR_WANDB_API_KEY' > /scratch/phalle.y/.wandb_key && chmod 600 /scratch/phalle.y/.wandb_key
# the SLURM scripts read this file into WANDB_API_KEY and point all wandb
# config/cache dirs at /scratch.
```

> **Version note:** TRL ↔ vLLM ↔ transformers are tightly coupled. If your env
> already has a working torch/vLLM, install the rest and align versions to it
> rather than forcing a torch reinstall. `vllm_mode: colocate` needs TRL ≥ 0.16.

The SLURM scripts already use `source activate /scratch/phalle.y/py310env`,
`cd /scratch/phalle.y/grpo-text-2-sql`, and redirect HF/torch/triton caches to
`/scratch`. If your env path or repo location differs, edit the top of each
`scripts/*.slurm`. Confirm the data paths in
[configs/grpo_bird_7b.yaml](configs/grpo_bird_7b.yaml) (training) and in
[scripts/evaluate.slurm](scripts/evaluate.slurm) (dev set).

Expected cluster paths (from your `/scratch/phalle.y` layout):

| Thing | Path |
|------|------|
| SFT adapter | `/scratch/phalle.y/bird_sft_adapter_7b/final_adapter` |
| BIRD train json | `/scratch/phalle.y/bird_train/train/train.json` |
| BIRD train DBs | `/scratch/phalle.y/bird_train/train/train_databases/train_databases` |
| BIRD dev json | `/scratch/phalle.y/bird_dev/dev_20240627/dev.json` |
| BIRD dev DBs | `/scratch/phalle.y/bird_dev/dev_20240627/dev_databases` |
| Merged SFT (created in step 1) | `/scratch/phalle.y/sft_merged_7b` |
| GRPO output | `/scratch/phalle.y/results_bird_grpo_7b` |

---

## Run the pipeline

### 0a. Build train.json from HuggingFace (login node, needs internet)
The BIRD train *questions* come from [`xu3kev/BIRD-SQL-data-train`](https://huggingface.co/datasets/xu3kev/BIRD-SQL-data-train)
(`db_id`, `question`, `evidence`, `SQL`, pre-extracted `schema`). Export it once
to a local JSON so the GPU jobs never need network access:

```bash
export HF_HOME=/scratch/phalle.y/hf_cache
python -m src.data.export_hf_to_json \
  --dataset xu3kev/BIRD-SQL-data-train \
  --out /scratch/phalle.y/bird_train/train/train.json
```
The local `train_databases/` are still required — the reward *executes* SQL
against them (the dataset only ships schema text, not DB files).

### 0b. Pre-flight (login node, no GPU)
Validates paths, schema loading, the executor, and the reward on real examples.
**Always run this** — it's free and catches 90% of "wasted a GPU job" bugs.

```bash
python scripts/check_setup.py \
  --train-json /scratch/phalle.y/bird_train/train/train.json \
  --db-root    /scratch/phalle.y/bird_train/train/train_databases/train_databases \
  --n 5
```
Expect `reward(gold)=1.00` and a lower `reward(wrong)` for each probe.

### 1. Merge the SFT adapter (once)
```bash
sbatch scripts/merge_sft.slurm        # → /scratch/phalle.y/sft_merged_7b
```

### 2. GRPO training
```bash
sbatch scripts/grpo_train.slurm       # → /scratch/phalle.y/results_bird_grpo_7b/final
```
Checkpoints every `save_steps`. The script passes `--resume`, so re-submitting
after a wall-clock timeout continues from the last checkpoint.

Quick functional run first (recommended): subset the data and cap steps to
confirm reward goes up before committing to a full run:
```bash
python -m src.train.grpo_train --config configs/grpo_bird_7b.yaml \
  --max-train-samples 256 --output-dir /scratch/phalle.y/grpo_smoke
```

### 3. Evaluate on BIRD dev
```bash
sbatch scripts/evaluate.slurm         # → results/grpo_dev_eval.json
```
Prints overall result accuracy + execution accuracy + a Simple/Moderate/
Challenging breakdown (comparable to the 46.9% SFT+DPO baseline).

---

## Weights & Biases

Enabled by default via the `wandb:` block in
[configs/grpo_bird_7b.yaml](configs/grpo_bird_7b.yaml):

```yaml
wandb:
  enabled: true
  project: bird-grpo-7b
  entity: null          # your wandb user/team
  run_name: grpo-bird-7b
```

- **Training** logs reward (mean + per-function), loss, KL, completion length,
  and sample completions (`log_completions: true`) live to the dashboard.
- **Eval** logs `dev/result_accuracy`, `dev/execution_accuracy`, and per-
  difficulty accuracy to the same project (`--wandb-project` in
  [scripts/evaluate.slurm](scripts/evaluate.slurm)).
- **Auth (HOME-quota-safe):** `wandb login` writes `~/.netrc` and fails when
  HOME is full. Instead drop your key in `/scratch/phalle.y/.wandb_key`; the
  SLURM scripts load it into `WANDB_API_KEY` and redirect `WANDB_DIR`,
  `WANDB_CACHE_DIR`, and `WANDB_CONFIG_DIR` to `/scratch`. If compute nodes have
  no internet, uncomment `export WANDB_MODE=offline` and run
  `wandb sync /scratch/phalle.y/wandb/<run>` later. Set `wandb.enabled: false`
  to disable entirely.

## The reward (`src/rewards/sql_reward.py`)

Layered, so even an all-wrong group yields a usable gradient early in training:

| Completion | Reward |
|-----------|-------:|
| Result set matches gold | **1.0** |
| Runs without error, wrong rows | 0.2 |
| Extractable SQL, errors on execution | 0.1 |
| No extractable SQL | 0.0 |

Set `reward.binary: true` in the config for a strict 0/1 reward. Weights
(`w_format`, `w_exec`) and the per-query `timeout` are configurable. Gold
queries are executed once and cached.

---

## Key design choices

- **Merge SFT, then fresh LoRA.** GRPO trains a new LoRA on the SFT-*merged*
  base instead of stacking on the SFT LoRA — cleaner vLLM weight syncing and a
  clean separation between SFT and RL deltas.
- **Same prompt everywhere.** [src/shared/prompt.py](src/shared/prompt.py) is
  the single template for dataset building, training, and eval. If your SFT
  used a different `evidence` format, fix it there and only there.
- **Prompts that overflow are dropped, not truncated.** With
  `max_prompt_length: 4096` on the H200, BIRD schema truncation (the main A10G
  bottleneck at 1536) is essentially gone; the few that still overflow are
  dropped so the model never trains on a half-schema.
- **vLLM colocate.** Training and rollout generation share the single H200;
  tune `vllm_gpu_memory_utilization` if you hit OOM.

---

## Tuning notes

- **OOM:** lower `vllm_gpu_memory_utilization` (e.g. 0.30), then
  `per_device_train_batch_size`, then `max_prompt_length`. Keep
  `global_batch (= per_device × grad_accum × n_gpus)` divisible by
  `num_generations`.
- **Reward collapses / policy degenerates:** raise `beta` (more KL anchoring to
  SFT) or lower `learning_rate`.
- **Reward flat at ~0.2:** model emits runnable-but-wrong SQL — check the prompt
  matches SFT, raise `max_completion_length`, or try `sample_rows: 2` so the
  schema shows example values.
```
