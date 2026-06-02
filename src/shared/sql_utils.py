"""SQL string extraction and light normalization.

The policy model is asked to "return only the SQL query", but during RL
exploration it will sometimes wrap output in markdown fences, add a
``<think>`` block (reasoning models), or prepend prose. ``extract_sql``
recovers the executable query from whatever the model emits so the reward
function gets a fair shot at executing it.
"""

from __future__ import annotations

import re

# Reasoning-style scratchpads that must be stripped before parsing.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Fenced code blocks: ```sql ... ``` or plain ``` ... ```
_FENCED = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Leading keyword that marks the start of a statement we care about.
_SQL_START = re.compile(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Best-effort extraction of a single SQL statement from model output.

    Returns the cleaned SQL string, or an empty string if nothing usable is
    found. Never raises — callers treat "" as a malformed completion.
    """
    if not text:
        return ""

    # 1. Drop reasoning scratchpads.
    text = _THINK_BLOCK.sub("", text)

    # 2. Prefer the contents of the first fenced code block, if present.
    fenced = _FENCED.search(text)
    if fenced:
        candidate = fenced.group(1).strip()
        if candidate:
            text = candidate

    # 3. Require an actual SQL statement: trim anything before the first SQL
    #    keyword. If none is present, the completion isn't SQL — return "".
    start = _SQL_START.search(text)
    if not start:
        return ""
    text = text[start.start():]

    # 4. Keep only up to the first statement terminator, collapse whitespace.
    text = text.split(";")[0].strip()
    text = re.sub(r"\s+", " ", text)
    return text
