"""Paired statistics for comparing two systems on the same questions.

Both systems answer the same questions, so every outcome is paired and the analysis uses that. The
functions take per-question 0/1 outcomes and return effect sizes with intervals; the test is reported
beside the interval, not instead of it.

* ``mcnemar``: exact McNemar test (a two-sided binomial test on the discordant pairs);
* ``paired_bootstrap_ci``: bootstrap interval of the difference, resampling questions as units;
* ``holm``: Holm adjustment of the p-values of one pre-declared family.

Standard library only, so no SciPy dependency. (The hallucination-rate accounting of the original design,
``har``, ``coverage`` and ``compare_systems``, lives in ``_archive/alzheimers_framework/evaluation/har_stats.py``.)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping


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


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values for the pre-declared primary comparisons."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = round(running, 6)
    return adjusted
