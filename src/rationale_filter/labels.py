"""The label rule of Figure 2 (Sohn et al., 2025, p.4), the two training tokens, and the threshold τ.

For one question and one snippet the answer is read twice, without and with the snippet.

    correct without   correct with   perplexity of the rationale lower with the snippet?   label
    yes               yes            yes                                                    helpful
    yes               yes            no                                                     discard
    yes               no             (not asked)                                            not helpful
    no                yes            (not asked)                                            helpful
    no                no             yes                                                    not helpful
    no                no             no                                                     discard

"Lower" means ΔPPL = PPL(x) − PPL(x, d) ≥ τ (Equation 3), so a difference equal to τ counts. τ is the top 25% of the
differences. How an answer that cannot be read counts is decided before this module is called (docs/methodology.md, D16);
here a correctness that is not a plain True or False is an error, never a label.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real

HELPFUL = "helpful"
NOT_HELPFUL = "not_helpful"
DISCARD = "discard"

TOKEN_HELPFUL = "[HELPFUL]"
TOKEN_NOT_HELPFUL = "[NOT_HELPFUL]"

TOP_FRACTION = 0.25  # the paper's choice, "fixed across all our experiments" (p.4)

_ROUNDING_GUARD = 1e-9  # so that 0.1 * 30 = 3.0000000000000004 still takes 3 values, not 4


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _perplexity(name: str, value: object) -> float:
    number = _finite(name, value)
    if number <= 0:
        raise ValueError(f"{name} must be above zero, got {value!r}")
    return number


def _correct(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be True or False, got {value!r}; an unreadable answer is settled before labelling")
    return value


def label_pair(*, correct_without: bool, correct_with: bool, ppl_without: float, ppl_with: float, tau: float) -> str:
    """``"helpful"``, ``"not_helpful"`` or ``"discard"`` for one question-snippet pair (the table in the module text)."""
    without = _correct("correct_without", correct_without)
    with_snippet = _correct("correct_with", correct_with)
    before = _perplexity("ppl_without", ppl_without)
    after = _perplexity("ppl_with", ppl_with)
    cut = _finite("tau", tau)
    if without and not with_snippet:
        return NOT_HELPFUL
    if with_snippet and not without:
        return HELPFUL
    lower = (before - after) >= cut
    if without:  # both correct
        return HELPFUL if lower else DISCARD
    return NOT_HELPFUL if lower else DISCARD  # both wrong


def to_training_token(label: str) -> str:
    """The token the filter is trained to produce. A discarded pair has none."""
    if label == HELPFUL:
        return TOKEN_HELPFUL
    if label == NOT_HELPFUL:
        return TOKEN_NOT_HELPFUL
    raise ValueError(f"only {HELPFUL!r} and {NOT_HELPFUL!r} have a training token, got {label!r}")


def perplexity_threshold(deltas: Iterable[float], top_fraction: float = TOP_FRACTION) -> float:
    """τ: the smallest difference that still lies in the top ``top_fraction`` of the pool.

    With n differences, the top ``ceil(n * top_fraction)`` (at least one) are kept and τ is the smallest of them, so a pool of
    4m values leaves exactly m at or above τ. Tied values at the boundary are all at or above τ. The pool is not changed.
    """
    fraction = _finite("top_fraction", top_fraction)
    if not 0 < fraction <= 1:
        raise ValueError(f"top_fraction must be above 0 and at most 1, got {top_fraction!r}")
    pool = sorted((_finite("a perplexity difference", value) for value in deltas), reverse=True)
    if not pool:
        raise ValueError("the pool of perplexity differences is empty")
    keep = max(1, math.ceil(len(pool) * fraction - _ROUNDING_GUARD))
    return pool[keep - 1]
