"""Paired statistics for comparing two systems on the same questions (docs/methodology.md, 7.9, decision D24).

Both systems answer the same questions, so every outcome is paired and the analysis uses that. The functions take per-question
0/1 outcomes (1 is a correct answer) and give effect sizes with intervals; the test is reported beside the interval.

    mcnemar              the exact McNemar test: a two-sided binomial test on the questions where the two systems differ
    paired_bootstrap_ci  a bootstrap interval of the difference in accuracy, resampling questions as units
    holm                 Holm's adjustment of the p-values of one family of comparisons declared before the first test run
    compare              one comparison: the counts, the p-value, and the difference with its interval

Adapted from ``_archive/rag2_adaptation_v1/experiments/medchange/stats.py`` (nine tests passed there on 2026-10-23); standard
library only. The archive is not changed.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class StatsError(ValueError):
    """An input of the analysis is malformed."""


def _paired(baseline: Mapping[str, Any], proposed: Mapping[str, Any]) -> tuple[str, ...]:
    shared = set(baseline) & set(proposed)
    if not shared:
        raise StatsError("no questions answered by both systems")
    if set(baseline) != set(proposed):
        raise StatsError(
            "the two systems did not answer the same question set; "
            f"baseline_only={sorted(set(baseline) - shared)[:5]} proposed_only={sorted(set(proposed) - shared)[:5]}"
        )
    return tuple(sorted(shared))


#: Up to this many trials the probabilities are computed directly. math.comb(n, k) * 0.5**n overflows or underflows a float a little
#: beyond it (a million questions would not be needed; MedQA's test set has 1,273), so larger counts are summed in log space.
_DIRECT_LIMIT = 1000


def _two_sided_direct(successes: int, n: int, p: float) -> float:
    probabilities = [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]
    observed = probabilities[successes]
    # A tolerance, because outcomes of equal probability must all be counted although their floating-point values differ a little.
    return min(1.0, math.fsum(q for q in probabilities if q <= observed * (1 + 1e-9)))


def _two_sided_log_space(successes: int, n: int, p: float) -> float:
    def log_probability(k: int) -> float:
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)

    logs = [log_probability(k) for k in range(n + 1)]
    observed = logs[successes]
    chosen = [value for value in logs if value <= observed + 1e-9]
    peak = max(chosen)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(value - peak) for value in chosen))


def binomial_two_sided_p(successes: int, n: int, p: float = 0.5) -> float:
    """The exact two-sided binomial p-value: the total probability of the outcomes that are no more likely than the observed one."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0 or isinstance(successes, bool) or not isinstance(successes, int) or not 0 <= successes <= n:
        raise StatsError(f"successes must be a whole number between 0 and n, got {successes!r} of {n!r}")
    if not 0 < p < 1:
        raise StatsError(f"p must be between 0 and 1, got {p!r}")
    if n == 0:
        return 1.0
    return _two_sided_direct(successes, n, p) if n <= _DIRECT_LIMIT else _two_sided_log_space(successes, n, p)


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


def _outcomes(name: str, outcomes: Mapping[str, int]) -> None:
    for key, value in outcomes.items():
        if isinstance(value, bool) or value not in (0, 1):
            raise StatsError(f"{name}: the outcome of {key} must be 0 or 1, got {value!r}")


def mcnemar(baseline: Mapping[str, int], proposed: Mapping[str, int]) -> McNemarResult:
    """Exact McNemar test on paired 0/1 outcomes. Only the questions where the systems differ carry information about the difference."""
    _outcomes("baseline", baseline)
    _outcomes("proposed", proposed)
    keys = _paired(baseline, proposed)
    baseline_only = sum(1 for k in keys if baseline[k] == 1 and proposed[k] == 0)
    proposed_only = sum(1 for k in keys if baseline[k] == 0 and proposed[k] == 1)
    both = sum(1 for k in keys if baseline[k] == 1 and proposed[k] == 1)
    neither = len(keys) - baseline_only - proposed_only - both
    discordant = baseline_only + proposed_only
    p = 1.0 if discordant == 0 else binomial_two_sided_p(proposed_only, discordant)
    return McNemarResult(len(keys), baseline_only, proposed_only, both, neither, p)


def paired_bootstrap_ci(
    baseline: Mapping[str, float],
    proposed: Mapping[str, float],
    *,
    seed: str = "bootstrap",
    iterations: int = 10000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """A bootstrap interval for mean(proposed) - mean(baseline).

    Questions are resampled as units, keeping the two outcomes of a question together; resampling the answers of each system
    on their own would throw the pairing away. The seed is text (the random generator of a text seed is the same in every
    process, unlike ``hash``), so the interval can be reproduced exactly.
    """
    if not 0 < confidence < 1:
        raise StatsError(f"confidence must be between 0 and 1, got {confidence!r}")
    if not isinstance(iterations, int) or iterations < 1:
        raise StatsError("iterations must be a positive whole number")
    keys = _paired(baseline, proposed)
    differences = [proposed[k] - baseline[k] for k in keys]
    n = len(differences)
    point = math.fsum(differences) / n
    rng = random.Random(seed)
    means = sorted(math.fsum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    low = means[int((1 - confidence) / 2 * iterations)]
    high = means[min(iterations - 1, int((1 + confidence) / 2 * iterations))]
    return {
        "n_pairs": n,
        "difference": round(point, 6),
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
        "confidence": confidence,
        "iterations": iterations,
    }


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values: the smallest is multiplied by the number of comparisons, the next by one less, and so on, never decreasing."""
    for name, p in p_values.items():
        if isinstance(p, bool) or not isinstance(p, int | float) or not 0 <= p <= 1:
            raise StatsError(f"the p-value of {name} must be between 0 and 1, got {p!r}")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for position, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (count - position) * p))
        adjusted[name] = round(running, 6)
    return adjusted


def compare(
    baseline: Mapping[str, int], proposed: Mapping[str, int], *, seed: str = "bootstrap", iterations: int = 10000
) -> dict[str, Any]:
    """The accuracy of both systems, the McNemar result, and the bootstrap interval of the difference, for one comparison."""
    test = mcnemar(baseline, proposed)
    interval = paired_bootstrap_ci(baseline, proposed, seed=seed, iterations=iterations)
    keys = _paired(baseline, proposed)
    return {
        "baseline_accuracy": round(sum(baseline[k] for k in keys) / len(keys), 6),
        "proposed_accuracy": round(sum(proposed[k] for k in keys) / len(keys), 6),
        "mcnemar": test.to_dict(),
        "difference": interval,
    }
