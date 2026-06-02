"""Execution-based reward for GRPO text-to-SQL.

GRPO needs a scalar reward per completion. We use a small, layered scheme so
the policy gets a usable gradient even early in training when *no* sample in a
group is fully correct (a pure 0/1 reward gives zero advantage for an all-wrong
group, stalling learning on hard BIRD questions):

    no extractable SQL          -> 0.0
    SQL extracted               -> + w_format   (default 0.1)
    SQL executes without error  -> + w_exec     (default 0.1)
    result set matches gold     ->   1.0        (correctness dominates)

So a syntactically valid but wrong query scores 0.2, an erroring query 0.1, and
a correct query 1.0. Pass ``binary=True`` for a strict 0/1 reward instead.

The factory returns a plain callable with the signature TRL expects
(``completions`` + dataset columns as kwargs) and a ``__name__`` for logging.
Gold-query results are cached so each gold SQL executes at most once.
"""

from __future__ import annotations

from typing import Callable

from src.shared.evaluator import results_match
from src.shared.schema_loader import db_path_for
from src.shared.sql_utils import extract_sql
from src.shared.sqlite_executor import execute_sql


def _completion_text(completion) -> str:
    """Normalize a TRL completion (conversational list or raw string) to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        # conversational format: [{"role": "assistant", "content": "..."}]
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "")
    return str(completion)


def make_sql_reward(
    db_root: str,
    timeout: float = 30.0,
    w_format: float = 0.1,
    w_exec: float = 0.1,
    correct_reward: float = 1.0,
    binary: bool = False,
) -> Callable:
    """Create the GRPO reward function bound to a database root directory."""

    gold_cache: dict[tuple[str, str], object] = {}

    def _gold_rows(db_file: str, gold_sql: str):
        key = (db_file, gold_sql)
        if key not in gold_cache:
            gold_cache[key] = execute_sql(db_file, gold_sql, timeout=timeout)
        return gold_cache[key]

    def sql_reward(completions, **kwargs) -> list[float]:
        db_ids = kwargs["db_id"]
        gold_sqls = kwargs["gold_sql"]
        rewards: list[float] = []

        for completion, db_id, gold_sql in zip(completions, db_ids, gold_sqls):
            pred_sql = extract_sql(_completion_text(completion))
            if not pred_sql:
                rewards.append(0.0)
                continue

            db_file = db_path_for(db_root, db_id)
            pred = execute_sql(db_file, pred_sql, timeout=timeout)
            gold = _gold_rows(db_file, gold_sql)

            if pred.success and gold.success and results_match(pred.rows, gold.rows):
                rewards.append(correct_reward)
                continue

            if binary:
                rewards.append(0.0)
                continue

            r = w_format  # SQL was extractable
            if pred.success:
                r += w_exec  # ...and ran without error
            rewards.append(r)

        return rewards

    # TRL logs reward functions by __name__.
    sql_reward.__name__ = "sql_exec_reward"
    return sql_reward
