"""GRPO training for BIRD text-to-SQL on Qwen2.5-Coder-7B.

Pipeline position: run *after* ``merge_adapter`` has produced an SFT-merged
base. This script trains a fresh GRPO LoRA on top of it, using vLLM for fast
group rollouts and an execution-based reward.

    python -m src.train.grpo_train --config configs/grpo_bird_7b.yaml

All hyperparameters live in the YAML; a few path-like fields accept CLI
overrides so the same config works across machines.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.data.build_dataset import build_dataset
from src.rewards.sql_reward import make_sql_reward


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Let a handful of CLI flags override the YAML (paths + quick-run knobs)."""
    if args.model_path:
        cfg["model"]["path"] = args.model_path
    if args.train_json:
        cfg["data"]["train_json"] = args.train_json
    if args.db_root:
        cfg["data"]["db_root"] = args.db_root
    if args.output_dir:
        cfg["grpo"]["output_dir"] = args.output_dir
    if args.max_train_samples is not None:
        cfg["data"]["max_train_samples"] = args.max_train_samples
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training for BIRD text-to-SQL.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", help="Override merged SFT model path")
    parser.add_argument("--train-json", help="Override BIRD train.json path")
    parser.add_argument("--db-root", help="Override train databases root")
    parser.add_argument("--output-dir", help="Override output dir")
    parser.add_argument("--max-train-samples", type=int,
                        help="Subset train set (quick runs)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint in output_dir")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    model_cfg, data_cfg = cfg["model"], cfg["data"]
    reward_cfg, lora_cfg, grpo_cfg = cfg["reward"], cfg["lora"], cfg["grpo"]

    model_path = model_cfg["path"]
    print(f"[grpo] policy model: {model_path}")

    # --- Weights & Biases (HF Trainer integration reads WANDB_PROJECT/_ENTITY) ---
    wb_cfg = cfg.get("wandb", {}) or {}
    report_to, run_name = "none", None
    if wb_cfg.get("enabled"):
        os.environ.setdefault("WANDB_PROJECT", wb_cfg.get("project", "bird-grpo-7b"))
        if wb_cfg.get("entity"):
            os.environ.setdefault("WANDB_ENTITY", wb_cfg["entity"])
        report_to, run_name = "wandb", wb_cfg.get("run_name")
        print(f"[grpo] wandb: project={os.environ['WANDB_PROJECT']} run={run_name}")

    # --- tokenizer (left padding is required for batched generation) ---
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- dataset: drop prompts that would overflow max_prompt_length ---
    train_ds = build_dataset(
        json_path=data_cfg["train_json"],
        db_root=data_cfg["db_root"],
        schema_source=data_cfg.get("schema_source", "auto"),
        tokenizer=tokenizer,
        max_prompt_tokens=grpo_cfg["max_prompt_length"],
        sample_rows=data_cfg.get("sample_rows", 0),
        limit=data_cfg.get("max_train_samples"),
    )

    # --- reward bound to the training databases ---
    reward_fn = make_sql_reward(
        db_root=data_cfg["db_root"],
        timeout=reward_cfg.get("timeout", 30.0),
        w_format=reward_cfg.get("w_format", 0.1),
        w_exec=reward_cfg.get("w_exec", 0.1),
        correct_reward=reward_cfg.get("correct_reward", 1.0),
        binary=reward_cfg.get("binary", False),
    )

    # --- fresh GRPO LoRA on top of the merged SFT model ---
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("alpha", 64),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=grpo_cfg["output_dir"],
        # optimization
        learning_rate=grpo_cfg.get("learning_rate", 1e-5),
        lr_scheduler_type=grpo_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=grpo_cfg.get("warmup_ratio", 0.03),
        num_train_epochs=grpo_cfg.get("num_train_epochs", 1),
        max_steps=grpo_cfg.get("max_steps", -1),
        per_device_train_batch_size=grpo_cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=grpo_cfg.get("gradient_accumulation_steps", 4),
        gradient_checkpointing=grpo_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=grpo_cfg.get("bf16", True),
        # GRPO rollout / loss
        # NOTE: TRL 1.5.1 has no `max_prompt_length` on GRPOConfig — we instead
        # pre-filter over-long prompts in build_dataset() (see max_prompt_tokens
        # below), so nothing is silently truncated.
        num_generations=grpo_cfg.get("num_generations", 8),
        max_completion_length=grpo_cfg.get("max_completion_length", 512),
        temperature=grpo_cfg.get("temperature", 0.9),
        top_p=grpo_cfg.get("top_p", 1.0),
        beta=grpo_cfg.get("beta", 0.04),
        scale_rewards=grpo_cfg.get("scale_rewards", True),
        # vLLM rollouts (colocate: one GPU shared by train + generate)
        use_vllm=grpo_cfg.get("use_vllm", True),
        vllm_mode=grpo_cfg.get("vllm_mode", "colocate"),
        vllm_gpu_memory_utilization=grpo_cfg.get("vllm_gpu_memory_utilization", 0.35),
        vllm_max_model_length=grpo_cfg.get(
            "vllm_max_model_length",
            grpo_cfg["max_prompt_length"] + grpo_cfg.get("max_completion_length", 512),
        ),
        # logging / checkpointing
        logging_steps=grpo_cfg.get("logging_steps", 1),
        save_steps=grpo_cfg.get("save_steps", 100),
        save_total_limit=grpo_cfg.get("save_total_limit", 3),
        log_completions=grpo_cfg.get("log_completions", True),
        report_to=report_to,
        run_name=run_name,
        seed=grpo_cfg.get("seed", 42),
    )

    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,   # TRL 1.5.1 arg name (was `tokenizer`)
        peft_config=peft_config,
    )

    resume = args.resume and any(Path(grpo_cfg["output_dir"]).glob("checkpoint-*"))
    print(f"[grpo] starting training (resume={resume}) ...")
    trainer.train(resume_from_checkpoint=resume)

    final_dir = str(Path(grpo_cfg["output_dir"]) / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[grpo] saved final GRPO adapter -> {final_dir}")


if __name__ == "__main__":
    main()
