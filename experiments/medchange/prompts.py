"""Answer prompt and verdict parsing (deterministic; no judge model).

The generator must open with ``VERDICT: <SUPPORTED|REFUTED|NOT ENOUGH INFORMATION>``
followed by a short justification. The primary outcome is that parsed verdict
versus the gold verdict, the same three-way task the MedChange authors used. An
answer that does not parse is counted wrong and reported separately. The same
template serves every arm; B0 simply has no evidence block. Publication dates are
NOT shown, so only admission (not in-context date reasoning) differs between arms.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION")
SYSTEM = ("You are a careful medical evidence assistant. You answer from the evidence "
          "you are given when there is any, and from your own knowledge otherwise.")

TEMPLATE = (
    "Question: {question}\n\n{evidence}"
    "Decide whether current medical evidence SUPPORTS the hypothesis in the question "
    "(the intervention or claim is effective or true), REFUTES it (ineffective, harmful "
    "or false), or provides NOT ENOUGH INFORMATION to decide.\n\n"
    "Start your answer with exactly one line of the form\n"
    "VERDICT: SUPPORTED\nVERDICT: REFUTED\nVERDICT: NOT ENOUGH INFORMATION\n"
    "(choose one), then justify it in at most three sentences."
)


def build_prompt(question: str, passages: Sequence[dict]) -> str:
    if passages:
        block = "Evidence:\n" + "\n".join(
            f"[{i}] {p['title']}. {p['abstract']}" for i, p in enumerate(passages, 1)) + "\n\n"
    else:
        block = ""
    return TEMPLATE.format(question=question.strip(), evidence=block)


_VERDICT = re.compile(r"VERDICT\s*:\s*\**\s*(NOT\s+ENOUGH\s+INFORMATION|SUPPORTED|REFUTED)",
                      re.IGNORECASE)


def parse_verdict(text: str) -> Optional[str]:
    """First ``VERDICT:`` line; None when absent (never guessed from the prose)."""
    m = _VERDICT.search(text or "")
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(1).upper())
