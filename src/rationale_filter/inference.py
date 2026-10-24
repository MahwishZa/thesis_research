"""The filter at answer time: its input, its decision, and keeping the helpful snippets.

The input is the template of the five released training examples (docs/methodology.md, section 5). The decision is the
softmax over the logits of the two label tokens at the first generated position, the larger winning; the authors' ``argmax``
picks index 0, the helpful token, on an exact tie. Comparing the two logits directly gives the same decision without the
rounding of a softmax. Each snippet is judged alone, from its own input (Paper p.9, limitations).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from numbers import Real
from typing import TypeVar

FILTER_INSTRUCTION = "Given the following evidence, determine whether it helps answer the provided question."

T = TypeVar("T")


def build_filter_input(*, evidence: str, question: str) -> str:
    """The text the filter reads: the instruction, the evidence and the question, with the outer whitespace of both removed."""
    for name, value in (("evidence", evidence), ("question", question)):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string, got {type(value).__name__}")
        if not value.strip():
            raise ValueError(f"the {name} is empty")
    return f"{FILTER_INSTRUCTION}\n\nEvidence: {evidence.strip()}\n\nQuestion: {question.strip()}"


def _logit(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def decide_helpful(*, logit_helpful: float, logit_not_helpful: float) -> bool:
    """True if the helpful token wins; an exact tie counts as helpful."""
    return _logit("logit_helpful", logit_helpful) >= _logit("logit_not_helpful", logit_not_helpful)


def judge_snippets(*, question: str, snippets: Sequence[str], judge: Callable[[str], bool]) -> list[bool]:
    """One decision per snippet, in order. ``judge`` gets the filter input of one snippet and says helpful or not.

    A snippet is judged from its own input only, so its decision does not depend on the other snippets. A failing judge
    raises; no snippet is ever given a default decision.
    """
    return [bool(judge(build_filter_input(evidence=snippet, question=question))) for snippet in snippets]


def decisions_for(*, question: str, snippets: Sequence[str], judge_many: Callable[[list[str]], Sequence[bool]]) -> list[bool]:
    """The same decisions as ``judge_snippets``, with one call for all the snippets of a question so that a model can batch them.

    ``judge_many`` gets the filter inputs of all the snippets, in order, and returns one decision for each. A snippet is still
    judged from its own input only; the batch changes nothing but speed. A count that does not match is an error.
    """
    inputs = [build_filter_input(evidence=snippet, question=question) for snippet in snippets]
    if not inputs:
        return []
    answers = list(judge_many(inputs))
    if len(answers) != len(inputs):
        raise ValueError(f"{len(inputs)} snippets but {len(answers)} decisions")
    return [bool(answer) for answer in answers]


def keep_helpful(items: Sequence[T], decisions: Sequence[bool]) -> list[T]:
    """The items whose decision is True, in their original order, as a new list of the very same objects."""
    if len(items) != len(decisions):
        raise ValueError(f"{len(items)} items but {len(decisions)} decisions")
    for decision in decisions:
        if not isinstance(decision, bool):
            raise TypeError(f"a decision must be True or False, got {decision!r}")
    return [item for item, decision in zip(items, decisions, strict=True) if decision]
