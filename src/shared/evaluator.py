"""Execution-based correctness: does the predicted SQL return the gold rows?

This is the same "result accuracy" metric used across the Spider/BIRD
experiments — two queries are equivalent iff their result *sets* match,
regardless of row order or the exact SQL text. It is the single source of
truth for both the GRPO reward and the dev-set evaluation.
"""

from __future__ import annotations

from .sqlite_executor import ExecResult, execute_sql


def _as_set(rows: list[tuple] | None) -> set:
    """Normalize a result set to an order-insensitive comparable form.

    Cells are stringified so that ``1`` (int) and ``1.0`` (float) coming from
    different queries don't spuriously mismatch on type alone, while still
    distinguishing genuinely different values.
    """
    if not rows:
        return set()
    return {tuple(str(c) for c in row) for row in rows}


def results_match(pred_rows: list[tuple] | None,
                  gold_rows: list[tuple] | None) -> bool:
    """True iff predicted and gold result sets are equal (order-insensitive)."""
    return _as_set(pred_rows) == _as_set(gold_rows)


def compare_sql(db_path: str, pred_sql: str, gold_sql: str,
                timeout: float = 30.0) -> tuple[bool, ExecResult, ExecResult]:
    """Execute both queries and report whether their result sets match.

    Returns ``(is_correct, pred_result, gold_result)`` so callers can also
    inspect *why* a prediction failed (parse error vs. wrong rows vs. timeout).
    """
    pred = execute_sql(db_path, pred_sql, timeout=timeout)
    gold = execute_sql(db_path, gold_sql, timeout=timeout)
    correct = pred.success and gold.success and results_match(pred.rows, gold.rows)
    return correct, pred, gold
