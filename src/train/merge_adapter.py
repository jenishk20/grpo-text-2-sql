"""Merge the BIRD SFT LoRA adapter into the base model.

GRPO starts from the SFT checkpoint. Rather than stack a second LoRA on top of
the SFT LoRA (which complicates vLLM rollouts and weight syncing), we bake the
SFT adapter into a full-precision model directory once, then train a brand-new
GRPO LoRA on top of *that* merged model.

    Qwen2.5-Coder-7B  +  bird_sft_adapter_7b  --merge-->  sft_merged_7b/
                                                              \\
                                                               +-- fresh GRPO LoRA

Run once before training:

    python -m src.train.merge_adapter \\
        --base   Qwen/Qwen2.5-Coder-7B-Instruct \\
        --adapter /scratch/phalle.y/bird_sft_adapter_7b \\
        --out    /scratch/phalle.y/sft_merged_7b
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(base: str, adapter: str, out: str, dtype: str = "bfloat16") -> None:
    torch_dtype = getattr(torch, dtype)
    print(f"[merge] loading base model: {base}")
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch_dtype, device_map="cpu"
    )

    print(f"[merge] applying LoRA adapter: {adapter}")
    model = PeftModel.from_pretrained(model, adapter)

    print("[merge] merge_and_unload ...")
    model = model.merge_and_unload()

    print(f"[merge] saving merged model -> {out}")
    model.save_pretrained(out, safe_serialization=True)

    # Tokenizer: prefer the adapter's (it may carry chat-template/pad tweaks
    # from SFT); fall back to the base model's.
    try:
        tok = AutoTokenizer.from_pretrained(adapter)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base)
    tok.save_pretrained(out)
    print("[merge] done.")


def _main() -> None:
    p = argparse.ArgumentParser(description="Merge SFT LoRA into the base model.")
    p.add_argument("--base", required=True, help="Base model id or path")
    p.add_argument("--adapter", required=True, help="SFT LoRA adapter dir")
    p.add_argument("--out", required=True, help="Output dir for merged model")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = p.parse_args()
    merge(args.base, args.adapter, args.out, args.dtype)


if __name__ == "__main__":
    _main()
