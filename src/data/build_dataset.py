"""Build a HuggingFace ``Dataset`` for GRPO from BIRD JSON.

Each example carries everything the trainer and reward need:

* ``prompt``    — chat-formatted user turn (TRL applies the chat template)
* ``db_id``     — used by the reward to locate the SQLite file
* ``gold_sql``  — reference query the reward executes for comparison
* plus question/evidence/difficulty metadata (handy for eval + analysis)

The same builder serves training (BIRD train split) and evaluation (BIRD dev
split) so the prompt construction is guaranteed identical on both sides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from src.shared.prompt import build_chat
from src.shared.schema_loader import db_path_for, load_schema


def load_records(json_path: str) -> list[dict[str, Any]]:
    """Read a BIRD-style JSON file (a flat list of question objects)."""
    with open(json_path) as f:
        return json.load(f)


def _gold_sql(record: dict) -> str:
    """BIRD stores gold as 'SQL'; Spider uses 'query'. Accept either."""
    return (record.get("SQL") or record.get("query") or "").strip()


def build_dataset(
    json_path: str,
    db_root: str,
    schema_source: str = "auto",
    tokenizer=None,
    max_prompt_tokens: int | None = None,
    sample_rows: int = 0,
    limit: int | None = None,
) -> Dataset:
    """Construct the GRPO dataset.

    ``schema_source`` controls where the schema in each prompt comes from:
      * ``auto``     — use the record's embedded ``schema`` field if present
                       (this matches the SFT formatting), else read the local DB
      * ``embedded`` — require the record's ``schema`` field; drop if missing
      * ``db``       — always read the schema from the local SQLite DB

    The local DB must exist either way, because the reward executes SQL against
    it; rows whose DB is missing are dropped.

    If ``tokenizer`` and ``max_prompt_tokens`` are given, examples whose rendered
    prompt exceeds the budget are dropped (rather than silently truncated
    mid-schema, which is what hurt the earlier A10G runs).
    """
    records = load_records(json_path)
    if limit:
        records = records[:limit]

    rows: list[dict[str, Any]] = []
    dropped_no_gold = dropped_too_long = dropped_no_db = dropped_no_schema = 0

    for i, rec in enumerate(records):
        db_id = rec["db_id"]
        gold = _gold_sql(rec)
        if not gold:
            dropped_no_gold += 1
            continue

        db_file = db_path_for(db_root, db_id)
        if not Path(db_file).exists():
            dropped_no_db += 1
            continue

        if schema_source == "db":
            schema = load_schema(db_file, sample_rows=sample_rows)
        else:  # auto | embedded
            schema = (rec.get("schema") or "").strip()
            if not schema:
                if schema_source == "embedded":
                    dropped_no_schema += 1
                    continue
                schema = load_schema(db_file, sample_rows=sample_rows)

        question = rec["question"]
        evidence = rec.get("evidence", "")
        chat = build_chat(schema, question, evidence)

        if tokenizer is not None and max_prompt_tokens is not None:
            n_tok = len(
                tokenizer.apply_chat_template(
                    chat, tokenize=True, add_generation_prompt=True
                )
            )
            if n_tok > max_prompt_tokens:
                dropped_too_long += 1
                continue

        rows.append(
            {
                "prompt": chat,
                "db_id": db_id,
                "gold_sql": gold,
                "question": question,
                "evidence": evidence,
                "difficulty": rec.get("difficulty", ""),
                "question_id": rec.get("question_id", i),
            }
        )

    print(
        f"[build_dataset] kept {len(rows)} / {len(records)} examples "
        f"(dropped: no_gold={dropped_no_gold}, missing_db={dropped_no_db}, "
        f"no_schema={dropped_no_schema}, too_long={dropped_too_long})"
    )
    return Dataset.from_list(rows)


def _main() -> None:
    p = argparse.ArgumentParser(description="Preview the BIRD GRPO dataset.")
    p.add_argument("--json", required=True, help="Path to BIRD train.json / dev.json")
    p.add_argument("--db-root", required=True, help="Directory of per-db folders")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--sample-rows", type=int, default=0)
    args = p.parse_args()

    ds = build_dataset(args.json, args.db_root, sample_rows=args.sample_rows,
                       limit=args.limit)
    print(f"\nColumns: {ds.column_names}\n")
    ex = ds[0]
    print("=== First prompt ===")
    print(ex["prompt"][0]["content"][:1500])
    print("\n=== Gold SQL ===")
    print(ex["gold_sql"])


if __name__ == "__main__":
    _main()
