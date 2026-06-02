"""Execute SQL against a SQLite database with a hard timeout.

The reward function runs untrusted, model-generated SQL hundreds of times per
training step. Two safety properties matter:

1. **Timeout** — a pathological query (cartesian join, no index) must not
   stall the whole training step. We arm a ``threading.Timer`` that calls
   ``Connection.interrupt()``, which aborts the running statement from another
   thread (the supported SQLite mechanism — not a thread kill).
2. **Read-only** — databases are opened ``mode=ro`` so a generated
   ``DELETE``/``UPDATE`` can never corrupt the benchmark DBs.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass


@dataclass
class ExecResult:
    success: bool
    rows: list[tuple] | None = None
    error: str | None = None

    @property
    def timed_out(self) -> bool:
        return self.error == "timeout"


def execute_sql(db_path: str, sql: str, timeout: float = 30.0,
                max_rows: int = 50_000) -> ExecResult:
    """Run ``sql`` against ``db_path`` read-only, aborting after ``timeout`` s.

    Returns an :class:`ExecResult`. Never raises — execution errors (bad SQL,
    missing columns, timeouts) are captured in ``error`` so the reward function
    can turn them into a low reward rather than crashing the trainer.
    """
    if not sql:
        return ExecResult(success=False, error="empty sql")

    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
    except sqlite3.Error as e:
        return ExecResult(success=False, error=f"connect failed: {e}")

    timer = threading.Timer(timeout, con.interrupt)
    timer.start()
    try:
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(max_rows)
        return ExecResult(success=True, rows=rows)
    except sqlite3.OperationalError as e:
        # interrupt() surfaces here as "interrupted"
        if "interrupt" in str(e).lower():
            return ExecResult(success=False, error="timeout")
        return ExecResult(success=False, error=str(e))
    except sqlite3.Error as e:
        return ExecResult(success=False, error=str(e))
    finally:
        timer.cancel()
        con.close()
