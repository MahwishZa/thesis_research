"""Perplexity of a rationale, with and without a snippet (Equations 3 and 4 of the paper; docs/methodology.md, decision D17).

    PPL = exp(-(1/L) * sum over the L tokens of log P(token | the tokens before it, the prompt))

Equation 4 scores the same text x, of the same length L, in both lines, once without and once with the document, so the
difference of the two perplexities is a difference in how well the prompt predicts one fixed text. The paper's text calls x
"the input query" and Figure 2 says "rationale"; the working default is the rationale the model wrote without evidence, scored
under the prompt without a snippet and under the prompt with it. A scorer is anything with
``token_logprobs(user_message, completion) -> list[float]``, the log-probability of each token of ``completion`` given the
prompt (see ``scorer.py`` for the llama.cpp one).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol


class TokenLogprobScorer(Protocol):
    def token_logprobs(self, user_message: str, completion: str) -> list[float]: ...


def perplexity(logprobs: Sequence[float]) -> float:
    """exp of the negative mean log-probability. A log-probability above zero, or not finite, is an error."""
    if len(logprobs) == 0:
        raise ValueError("a perplexity needs at least one token")
    for position, value in enumerate(logprobs):
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value > 1e-9:
            raise ValueError(f"token {position}: a log-probability must be a finite number up to 0, got {value!r}")
    return math.exp(-math.fsum(logprobs) / len(logprobs))


def rationale_perplexities(
    scorer: TokenLogprobScorer, *, prompt_without: str, prompt_with: str, rationale: str
) -> tuple[float, float]:
    """``(PPL(x), PPL(x, d))`` of the rationale under the prompt without and the prompt with the snippet.

    Both prompts are given as the user message of one turn; the rationale is scored as the answer to it. The two scorings
    read the same tokens, so their lengths agree; a scorer that returns different lengths is refused.
    """
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("the rationale to score is empty")
    without = scorer.token_logprobs(prompt_without, rationale)
    with_snippet = scorer.token_logprobs(prompt_with, rationale)
    if len(without) != len(with_snippet):
        raise ValueError(f"the rationale has {len(without)} tokens without the snippet and {len(with_snippet)} with it")
    return perplexity(without), perplexity(with_snippet)
