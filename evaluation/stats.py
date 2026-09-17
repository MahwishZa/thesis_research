"""Paired analysis for the baseline / proposed comparison.

Both systems answer the same questions, so every outcome is paired and the
analysis should use that. The functions here take per-question outcomes and
return effect sizes with intervals; the test is reported beside the interval,
not instead of it.

Implemented with the standard library only - the exact McNemar test is a
two-sided binomial test on the discordant pairs, which is a few lines and
avoids a SciPy dependency the environment does not currently have.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


class StatsError(ValueError):
    """Raised when an analysis input is malformed."""


def _paired(baseline: Mapping[str, int],
            proposed: Mapping[str, int]) -> tuple[str, ...]:
    shared = set(baseline) & set(proposed)
    if not shared:
        raise StatsError("no questions answered by both systems")
    if set(baseline) != set(proposed):
        raise StatsError(
            "the two systems did not answer the same question set; "
            f"baseline_only={sorted(set(baseline)-shared)} "
            f"proposed_only={sorted(set(proposed)-shared)}"
        )
    return tuple(sorted(shared))


def binomial_two_sided_p(successes: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value by summing outcomes no more likely."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]
    observed = probs[successes]
    # Floating tolerance: outcomes of equal probability must all be counted.
    return min(1.0, sum(q for q in probs if q <= observed * (1 + 1e-9)))


@dataclass(frozen=True)
class McNemarResult:
    n_pairs: int
    baseline_only: int
    proposed_only: int
    both: int
    neither: int
    p_value: float

    @property
    def discordant(self) -> int:
        return self.baseline_only + self.proposed_only

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "baseline_only": self.baseline_only,
            "proposed_only": self.proposed_only,
            "both": self.both,
            "neither": self.neither,
            "discordant": self.discordant,
            "p_value": round(self.p_value, 6),
        }


def mcnemar(baseline: Mapping[str, int],
            proposed: Mapping[str, int]) -> McNemarResult:
    """Exact McNemar test on paired binary outcomes.

    ``baseline_only`` counts questions where the baseline shows the outcome
    (e.g. hallucinated) and the proposed system does not. Concordant pairs
    carry no information about the difference, which is why the test uses only
    the discordant ones.
    """
    keys = _paired(baseline, proposed)
    b_only = sum(1 for k in keys if baseline[k] == 1 and proposed[k] == 0)
    p_only = sum(1 for k in keys if baseline[k] == 0 and proposed[k] == 1)
    both = sum(1 for k in keys if baseline[k] == 1 and proposed[k] == 1)
    neither = sum(1 for k in keys if baseline[k] == 0 and proposed[k] == 0)

    discordant = b_only + p_only
    p = 1.0 if discordant == 0 else binomial_two_sided_p(p_only, discordant)
    return McNemarResult(len(keys), b_only, p_only, both, neither, p)


def paired_bootstrap_ci(
    baseline: Mapping[str, float],
    proposed: Mapping[str, float],
    *,
    seed: str = "bootstrap",
    iterations: int = 10000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap CI for mean(proposed) - mean(baseline).

    Questions are resampled as units, keeping each question's two outcomes
    together. That is what makes the interval respect the pairing; resampling
    answers independently would discard it.
    """
    keys = _paired({k: 0 for k in baseline}, {k: 0 for k in proposed})
    diffs = [proposed[k] - baseline[k] for k in keys]
    n = len(diffs)
    point = sum(diffs) / n

    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        means.append(
            sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        )
    means.sort()
    lo = means[int((1 - confidence) / 2 * iterations)]
    hi = means[min(iterations - 1,
                   int((1 + confidence) / 2 * iterations))]
    return {
        "n_pairs": n,
        "difference": round(point, 6),
        "ci_low": round(lo, 6),
        "ci_high": round(hi, 6),
        "confidence": confidence,
        "iterations": iterations,
    }


def har(
    hallucinated: Mapping[str, int],
    abstained: Optional[Mapping[str, int]] = None,
) -> dict[str, Any]:
    """Hallucinated Answer Rate, reported both ways.

    An abstention contains no claims and so can never be labelled
    hallucinated. Reporting only the conditional rate therefore rewards a
    system for declining to answer. Both denominators are returned, always
    with the abstention rate, so a rate can never be read without it.
    """
    n_all = len(hallucinated)
    if n_all == 0:
        raise StatsError("no answers to score")
    abstained = abstained or {k: 0 for k in hallucinated}

    answered = [k for k in hallucinated if not abstained.get(k, 0)]
    n_answered = len(answered)
    total_h = sum(hallucinated.values())

    return {
        "n_answers": n_all,
        "n_answered": n_answered,
        "n_abstained": n_all - n_answered,
        "abstention_rate": round((n_all - n_answered) / n_all, 6),
        "n_hallucinated": total_h,
        # Denominator = answers actually produced.
        "har_conditional": (
            None if n_answered == 0
            else round(sum(hallucinated[k] for k in answered) / n_answered, 6)
        ),
        # Denominator = every item; abstentions count as non-hallucinated.
        "har_all_items": round(total_h / n_all, 6),
    }


def subtype_distribution(
    subtypes: Sequence[Optional[str]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in subtypes:
        if s:
            counts[s] = counts.get(s, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def outcome_crosstab(
    baseline: Mapping[str, int],
    proposed: Mapping[str, int],
) -> dict[str, list[str]]:
    """Which questions fall in each cell, for error analysis.

    Returns question ids rather than counts so the write-up can pull
    representative examples - including the cases where the proposed system
    did worse, which are the ones most easily left out.
    """
    keys = _paired(baseline, proposed)
    return {
        "baseline_only": [k for k in keys
                          if baseline[k] == 1 and proposed[k] == 0],
        "proposed_only": [k for k in keys
                          if baseline[k] == 0 and proposed[k] == 1],
        "both": [k for k in keys if baseline[k] == 1 and proposed[k] == 1],
        "neither": [k for k in keys if baseline[k] == 0 and proposed[k] == 0],
    }


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values for the pre-declared primary comparisons."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = round(running, 6)
    return adjusted


@dataclass(frozen=True)
class Coverage:
    """Answer accounting for one system, kept separate from any rate.

    HAR alone is not interpretable: a system that answers nothing has no
    hallucinated answers. Every HAR must be read next to these counts, which
    is why they travel together rather than being computed on demand.
    """

    total_questions: int
    answered: int
    abstained: int
    hallucinated: int
    non_hallucinated: int

    @property
    def answer_coverage(self) -> float:
        return self.answered / self.total_questions if self.total_questions else 0.0

    @property
    def is_degenerate(self) -> bool:
        """True when the system answered nothing, so no rate is meaningful."""
        return self.answered == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "answered": self.answered,
            "abstained": self.abstained,
            "hallucinated": self.hallucinated,
            "non_hallucinated": self.non_hallucinated,
            "answer_coverage": round(self.answer_coverage, 6),
            "is_degenerate": self.is_degenerate,
        }


def coverage(
    hallucinated: Mapping[str, int],
    abstained: Mapping[str, int],
) -> Coverage:
    """Count answered, abstained and hallucinated outcomes for one system."""
    if set(hallucinated) != set(abstained):
        raise StatsError(
            "hallucination and abstention records cover different questions"
        )
    total = len(hallucinated)
    answered_keys = [k for k in hallucinated if not abstained[k]]
    for k in hallucinated:
        if abstained[k] and hallucinated[k]:
            raise StatsError(
                f"{k}: an abstention cannot also be hallucinated"
            )
    h = sum(hallucinated[k] for k in answered_keys)
    return Coverage(
        total_questions=total,
        answered=len(answered_keys),
        abstained=total - len(answered_keys),
        hallucinated=h,
        non_hallucinated=len(answered_keys) - h,
    )


def compare_systems(
    baseline_hallucinated: Mapping[str, int],
    baseline_abstained: Mapping[str, int],
    proposed_hallucinated: Mapping[str, int],
    proposed_abstained: Mapping[str, int],
    *,
    seed: str = "har",
    iterations: int = 10000,
) -> dict[str, Any]:
    """The primary comparison, with the accounting that makes it readable.

    Refuses to emit a headline difference when either system answered
    nothing, because a rate over zero answers is not a rate. The refusal is a
    recorded field rather than an exception: a degenerate run is a result the
    thesis should report, not a crash.
    """
    base_cov = coverage(baseline_hallucinated, baseline_abstained)
    prop_cov = coverage(proposed_hallucinated, proposed_abstained)

    base_har = har(baseline_hallucinated, baseline_abstained)
    prop_har = har(proposed_hallucinated, proposed_abstained)

    degenerate = base_cov.is_degenerate or prop_cov.is_degenerate

    report: dict[str, Any] = {
        "baseline": {"coverage": base_cov.to_dict(), "har": base_har},
        "proposed": {"coverage": prop_cov.to_dict(), "har": prop_har},
        "coverage_difference": round(
            prop_cov.answer_coverage - base_cov.answer_coverage, 6
        ),
        "interpretable": not degenerate,
    }

    if degenerate:
        report["warning"] = (
            "at least one system answered no questions; a hallucination rate "
            "over zero answers is undefined and this comparison must not be "
            "reported as hallucination reduction"
        )
        return report

    # Abstentions count as non-hallucinated here, which is the denominator a
    # coverage-losing system cannot game: it is reported beside the coverage.
    report["mcnemar_all_items"] = mcnemar(
        baseline_hallucinated, proposed_hallucinated
    ).to_dict()
    report["delta_har_all_items"] = paired_bootstrap_ci(
        {k: float(v) for k, v in baseline_hallucinated.items()},
        {k: float(v) for k, v in proposed_hallucinated.items()},
        seed=seed, iterations=iterations,
    )
    return report
