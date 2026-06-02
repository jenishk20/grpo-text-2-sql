"""Evaluate a policy on the BIRD dev set (execution / result accuracy).

Reuses the exact prompt builder and execution metric from training, so the
number reported here is directly comparable to the GRPO reward and to the
46.9% SFT+DPO baseline.

Greedy decoding (temperature 0) by default. Two backends:

    --backend vllm   (default, fast)   model = merged full model;
                                        --adapter applies a LoRA via vLLM
    --backend hf     (robust fallback) transformers + peft

Examples
--------
    # GRPO adapter on top of the SFT-merged base
    python -m src.eval.evaluate \\
        --model /scratch/phalle.y/sft_merged_7b \\
        --adapter /scratch/phalle.y/results_bird_dpo_7b/grpo/final \\
        --dev-json /scratch/phalle.y/bird_dev/dev.json \\
        --db-root  /scratch/phalle.y/bird_dev/dev_databases \\
        --out results/grpo_dev_eval.json

    # SFT baseline only (no adapter): merge SFT first, point --model at it
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer

from src.data.build_dataset import build_dataset
from src.shared.evaluator import results_match
from src.shared.schema_loader import db_path_for
from src.shared.sql_utils import extract_sql
from src.shared.sqlite_executor import execute_sql


def render_prompts(dataset, tokenizer) -> list[str]:
    """Apply the chat template to each example's prompt messages."""
    return [
        tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True
        )
        for ex in dataset
    ]


def generate_vllm(model: str, adapter: str | None, prompts: list[str],
                  max_tokens: int, max_model_len: int) -> list[str]:
    from vllm import LLM, SamplingParams

    enable_lora = adapter is not None
    llm = LLM(
        model=model,
        dtype="bfloat16",
        enable_lora=enable_lora,
        max_lora_rank=64,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    lora_request = None
    if enable_lora:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("grpo", 1, adapter)

    outputs = llm.generate(prompts, sampling, lora_request=lora_request)
    return [o.outputs[0].text for o in outputs]


def generate_hf(model: str, adapter: str | None, prompts: list[str],
                max_tokens: int, batch_size: int = 16) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter:
        lm = PeftModel.from_pretrained(lm, adapter)
    lm.eval()

    texts: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True).to(lm.device)
        with torch.no_grad():
            out = lm.generate(**enc, max_new_tokens=max_tokens, do_sample=False)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"[eval/hf] generated {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return texts


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a policy on BIRD dev.")
    p.add_argument("--model", required=True, help="Full model path (merged base)")
    p.add_argument("--adapter", default=None, help="Optional GRPO LoRA adapter dir")
    p.add_argument("--dev-json", required=True)
    p.add_argument("--db-root", required=True)
    p.add_argument("--out", default="results/dev_eval.json")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--limit", type=int, default=None, help="Eval first N (smoke test)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--wandb-project", default=None,
                   help="If set, log the dev metrics to this W&B project")
    p.add_argument("--wandb-run", default=None)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model)
    dataset = build_dataset(args.dev_json, args.db_root, limit=args.limit)
    prompts = render_prompts(dataset, tokenizer)

    print(f"[eval] generating {len(prompts)} completions via {args.backend} ...")
    if args.backend == "vllm":
        completions = generate_vllm(args.model, args.adapter, prompts,
                                    args.max_tokens, args.max_model_len)
    else:
        completions = generate_hf(args.model, args.adapter, prompts, args.max_tokens)

    by_diff_total: dict[str, int] = defaultdict(int)
    by_diff_correct: dict[str, int] = defaultdict(int)
    n_correct = n_exec = 0
    records = []

    for ex, raw in zip(dataset, completions):
        pred_sql = extract_sql(raw)
        db_file = db_path_for(args.db_root, ex["db_id"])
        pred = execute_sql(db_file, pred_sql, timeout=args.timeout)
        gold = execute_sql(db_file, ex["gold_sql"], timeout=args.timeout)
        correct = pred.success and gold.success and results_match(pred.rows, gold.rows)

        n_exec += int(pred.success)
        n_correct += int(correct)
        diff = ex.get("difficulty") or "unknown"
        by_diff_total[diff] += 1
        by_diff_correct[diff] += int(correct)

        records.append({
            "question_id": ex["question_id"],
            "db_id": ex["db_id"],
            "question": ex["question"],
            "gold_sql": ex["gold_sql"],
            "pred_sql": pred_sql,
            "correct": correct,
            "exec_ok": pred.success,
            "difficulty": diff,
            "error": pred.error,
        })

    total = len(dataset)
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "total": total,
        "result_accuracy": round(n_correct / total, 4) if total else 0.0,
        "execution_accuracy": round(n_exec / total, 4) if total else 0.0,
        "by_difficulty": {
            d: {
                "accuracy": round(by_diff_correct[d] / by_diff_total[d], 4),
                "count": by_diff_total[d],
            }
            for d in sorted(by_diff_total)
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    print("\n=== BIRD dev evaluation ===")
    print(f"Result accuracy   : {summary['result_accuracy']:.1%} "
          f"({n_correct}/{total})")
    print(f"Execution accuracy: {summary['execution_accuracy']:.1%}")
    for d, s in summary["by_difficulty"].items():
        print(f"  {d:<12}: {s['accuracy']:.1%} ({s['count']})")
    print(f"\nWrote {out_path}")

    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=summary)
        metrics = {
            "dev/result_accuracy": summary["result_accuracy"],
            "dev/execution_accuracy": summary["execution_accuracy"],
        }
        for d, s in summary["by_difficulty"].items():
            metrics[f"dev/accuracy_{d}"] = s["accuracy"]
        wandb.log(metrics)
        wandb.finish()
        print("Logged dev metrics to W&B.")


if __name__ == "__main__":
    main()
