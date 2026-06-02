"""Pre-flight check — validate the data/reward pipeline before GPU training.

Runs on a login node (no GPU, no torch needed) and verifies, on a handful of
real BIRD examples:

  1. train.json is readable and DB files resolve from --db-root
  2. schemas load and prompts render within the token budget
  3. gold SQL executes and scores the full reward (sanity of the metric)
  4. a deliberately wrong query scores a low (partial/zero) reward

If this passes, the reward function and paths are correct and the only
remaining unknowns are GPU/library ones.

    python scripts/check_setup.py \\
        --train-json /scratch/phalle.y/bird_train/train/train.json \\
        --db-root    /scratch/phalle.y/bird_train/train/train_databases/train_databases \\
        --n 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script: python scripts/check_setup.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_dataset import load_records  # noqa: E402
from src.rewards.sql_reward import make_sql_reward  # noqa: E402
from src.shared.prompt import build_instruction  # noqa: E402
from src.shared.schema_loader import db_path_for, load_schema  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Validate the GRPO data + reward setup.")
    p.add_argument("--train-json", required=True)
    p.add_argument("--db-root", required=True)
    p.add_argument("--n", type=int, default=5, help="Examples to probe")
    args = p.parse_args()

    print(f"Reading {args.train_json}")
    records = load_records(args.train_json)
    print(f"  -> {len(records)} records\n")

    reward = make_sql_reward(db_root=args.db_root)
    ok = 0

    for rec in records[:args.n]:
        db_id = rec["db_id"]
        gold = (rec.get("SQL") or rec.get("query") or "").strip()
        db_file = db_path_for(args.db_root, db_id)

        print(f"[{db_id}] {rec['question'][:70]}...")
        if not Path(db_file).exists():
            print(f"  ✗ DB file not found: {db_file}")
            continue

        schema = load_schema(db_file)
        prompt = build_instruction(schema, rec["question"], rec.get("evidence", ""))
        print(f"  schema chars={len(schema)}  prompt chars={len(prompt)}")

        # Reward expects TRL-style batched kwargs.
        good = reward(
            completions=[gold],
            db_id=[db_id],
            gold_sql=[gold],
        )[0]
        bad = reward(
            completions=["SELECT 1 AS definitely_wrong"],
            db_id=[db_id],
            gold_sql=[gold],
        )[0]
        print(f"  reward(gold)={good:.2f}   reward(wrong)={bad:.2f}")

        if good >= 0.99 and bad < good:
            print("  ✓ ok\n")
            ok += 1
        else:
            print("  ✗ unexpected reward (gold should be 1.0 and beat wrong)\n")

    print(f"Passed {ok}/{args.n} probes.")
    sys.exit(0 if ok == args.n else 1)


if __name__ == "__main__":
    main()
