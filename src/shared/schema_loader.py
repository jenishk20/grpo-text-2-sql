"""Load a SQLite database schema as a DDL string for prompting.

We read the ``CREATE TABLE`` statements straight out of ``sqlite_master`` so
the schema shown to the model is exactly the one the gold SQL was written
against. Optionally appends a few sample rows per table, which measurably
helps text-to-SQL models pick the right columns/value formats.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path


def _connect_ro(db_path: str) -> sqlite3.Connection:
    """Open the database read-only so schema loading can never mutate it."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def db_path_for(db_root: str, db_id: str) -> str:
    """Resolve the conventional BIRD/Spider path: {root}/{db_id}/{db_id}.sqlite."""
    return str(Path(db_root) / db_id / f"{db_id}.sqlite")


@lru_cache(maxsize=512)
def load_schema(db_path: str, sample_rows: int = 0) -> str:
    """Return the schema of ``db_path`` as concatenated DDL.

    ``sample_rows`` > 0 appends that many example rows per table as a SQL
    comment. Results are cached per (db_path, sample_rows) since a database's
    schema is reused across every question that targets it.
    """
    con = _connect_ro(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY name"
        )
        tables = cur.fetchall()

        blocks: list[str] = []
        for name, ddl in tables:
            block = ddl.strip() + ";"
            if sample_rows > 0:
                block += "\n" + _format_samples(cur, name, sample_rows)
            blocks.append(block)
        return "\n\n".join(blocks)
    finally:
        con.close()


def _format_samples(cur: sqlite3.Cursor, table: str, n: int) -> str:
    """Render up to ``n`` rows of ``table`` as a SQL comment block."""
    try:
        cur.execute(f'SELECT * FROM "{table}" LIMIT {n}')
        rows = cur.fetchall()
        if not rows:
            return f"/* {table}: (empty) */"
        cols = [d[0] for d in cur.description]
        lines = [f"/* {n} sample rows from {table}:", "   " + " | ".join(cols)]
        for row in rows:
            lines.append("   " + " | ".join(str(v) for v in row))
        lines.append("*/")
        return "\n".join(lines)
    except sqlite3.Error:
        return f"/* {table}: (sample unavailable) */"
