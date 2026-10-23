"""The prompt that answers a question, with or without evidence (docs/methodology.md, decision D14).

The paper prints the prompt for the question only (p.4) and no template for answering with evidence; the README of the released
code says "prompt your LLM with the question plus the filtered evidence". The working default here keeps the paper's wording
and puts the kept snippets in an "Evidence" section before the question, one paragraph per snippet, in the order the list has:

    The following are multiple choice questions about medical knowledge. ... as the final answer.
    Evidence:
    <snippet 1>

    <snippet 2>
    Here is the question: <question with its options inline>

Without snippets, and also when the filter keeps none, the prompt is exactly the rationale prompt of Part 1, so the answer
without retrieval is the final option of the rationale itself (the rationale prompt already asks for one).
"""

from __future__ import annotations

from collections.abc import Sequence

from rationale_query.prompt import RATIONALE_PROMPT, build_rationale_prompt

_QUESTION_MARKER = "Here is the question: "
INSTRUCTION = RATIONALE_PROMPT[: RATIONALE_PROMPT.index(_QUESTION_MARKER)]
EVIDENCE_HEADING = "Evidence:\n"


def build_answer_prompt(question: str, snippets: Sequence[str] = ()) -> str:
    """The prompt for one question. ``snippets`` are the kept snippets in order; none gives the plain rationale prompt."""
    if isinstance(snippets, str):
        raise TypeError("snippets must be a list of texts, not one text")
    kept = []
    for position, snippet in enumerate(snippets):
        if not isinstance(snippet, str) or not snippet.strip():
            raise ValueError(f"snippet {position} is empty or not text")
        kept.append(snippet.strip())
    plain = build_rationale_prompt(question)  # also checks the question
    if not kept:
        return plain
    return INSTRUCTION + EVIDENCE_HEADING + "\n\n".join(kept) + "\n" + plain[len(INSTRUCTION) :]
