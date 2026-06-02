"""The single text-to-SQL prompt template, shared by training and inference.

CRITICAL: this must match the instruction format the SFT adapter was trained
on. GRPO continues from that SFT checkpoint, so any drift between the SFT
prompt and the GRPO/eval prompt puts the policy off-distribution and wastes the
adapter. If your SFT data formatted ``evidence`` differently, change it *here*
and nowhere else.
"""

from __future__ import annotations

INSTRUCTION_TEMPLATE = """Convert the following natural language question into a valid SQL query.

Database Schema:
{schema}

Question: {question}{evidence_block}

Return only the SQL query with no explanation."""


def build_instruction(schema: str, question: str, evidence: str | None = None) -> str:
    """Render the user instruction. BIRD's ``evidence`` is appended when present."""
    evidence_block = ""
    if evidence and evidence.strip():
        evidence_block = f"\n\nEvidence: {evidence.strip()}"
    return INSTRUCTION_TEMPLATE.format(
        schema=schema,
        question=question.strip(),
        evidence_block=evidence_block,
    )


def build_chat(schema: str, question: str, evidence: str | None = None) -> list[dict]:
    """Wrap the instruction as a chat message list.

    TRL/transformers applies the model's chat template to this, reproducing the
    exact Qwen chat formatting used during SFT.
    """
    return [{"role": "user", "content": build_instruction(schema, question, evidence)}]
