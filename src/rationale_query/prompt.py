"""The chain-of-thought prompt of the paper (Sohn et al., 2025, section 3.3, p.4; taken from Kim et al., 2024)."""

from __future__ import annotations

#: The text of the box on p.4, ending where "[initial_query]" stands. The paper prints a line break before
#: "Here is the question:"; whether a single or a double line break was sent is not stated (an open item).
RATIONALE_PROMPT = (
    "The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, "
    "starting by summarizing the available information. Output your explanation and single option from the given "
    "options as the final answer.\n"
    "Here is the question: "
)


def build_rationale_prompt(question: str) -> str:
    """The prompt for one question. The question text carries its options inline, as in the released samples."""
    if not isinstance(question, str):
        raise TypeError(f"the question must be a string, got {type(question).__name__}")
    text = question.strip()
    if not text:
        raise ValueError("the question is empty")
    return RATIONALE_PROMPT + text
