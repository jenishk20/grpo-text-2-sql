"""Export a HuggingFace BIRD dataset to a local BIRD-style train.json.

Run ONCE on the login node (it needs internet). Training then reads the local
JSON, so GPU compute nodes — which usually have no network — never call the Hub.

Preserves the pre-extracted ``schema`` field so GRPO prompts can match the exact
formatting the SFT adapter was trained on.

    export HF_HOME=/scratch/phalle.y/hf_cache
    python -m src.data.export_hf_to_json \\
        --dataset xu3kev/BIRD-SQL-data-train \\
        --out /scratch/phalle.y/bird_train/train/train.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Export a HF BIRD dataset to train.json.")
    p.add_argument("--dataset", default="xu3kev/BIRD-SQL-data-train")
    p.add_argument("--split", default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    print(f"[export] loading {args.dataset} (split={args.split})")
    ds = load_dataset(args.dataset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    records = []
    n_no_gold = 0
    for i, ex in enumerate(ds):
        gold = (ex.get("SQL") or ex.get("query") or "").strip()
        if not gold:
            n_no_gold += 1
            continue
        records.append(
            {
                "question_id": i,
                "db_id": ex["db_id"],
                "question": ex["question"],
                "evidence": ex.get("evidence", "") or "",
                "SQL": gold,
                "schema": ex.get("schema", "") or "",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(records, f)
    print(f"[export] wrote {len(records)} records -> {out} "
          f"(skipped {n_no_gold} with no gold SQL)")


if __name__ == "__main__":
    main()
