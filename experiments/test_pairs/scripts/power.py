"""Sample size for the paired bias probe (specification 43).

The primary outcome is paired and categorical: for each pair, was the older
passage admitted and the newer one not, or the reverse? Concordant pairs
(both admitted, or neither) carry no information about asymmetry, so the
analysis reduces to McNemar's test over the discordant pairs, and the sample
size follows from two quantities the pilot must supply:

* ``discordant_rate`` - the share of pairs on which the two passages receive
  different admission decisions;
* ``older_share`` - among those, the share favouring the older passage
  (0.5 is exactly no bias).

Both are estimates, and nothing here invents them: :func:`required_pairs`
raises unless they are measured.

Sizing is **exact**, by enumerating the binomial distribution, because the
analysis it sizes for is exact. The discordant count here is at most a few
hundred, so enumeration costs nothing, and matching the two matters: a normal
approximation sized for 80% power delivers roughly 77% against the exact
two-sided binomial test at these counts, which would leave the thesis
quietly under-powered.

The result is a *floor*, not a target: it counts usable pairs after
attrition, so the number of candidates to construct is this figure divided by
the pilot's yield rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


#: Conventional levels only; an unusual alpha or power should be justified,
#: not silently supported.
_ALPHAS = (0.05, 0.01)
_POWERS = (0.80, 0.90, 0.95)

#: Enumeration ceiling. Larger pools are not refused, but the search for a
#: required sample size stops here rather than running away.
_MAX_DISCORDANT = 100_000


def _critical_value(discordant: int, alpha: float) -> Optional[int]:
    """Largest k whose two-sided exact binomial p-value is at most alpha.

    Under the null the discordant pairs split evenly, so the test statistic
    is Binomial(discordant, 0.5) and the rejection region is the pair of
    tails {k <= c} and {k >= discordant - c}. Returns None when no rejection
    region of size alpha exists, which happens for very small counts.
    """

    total = 2.0 ** discordant

    cumulative = 0.0
    critical: Optional[int] = None

    for k in range(discordant // 2 + 1):
        cumulative += math.comb(discordant, k) / total
        if 2.0 * cumulative <= alpha:
            critical = k
        else:
            break

    return critical


def exact_power(discordant: int, share: float, alpha: float) -> float:
    """Power of the two-sided exact binomial test at the given share."""

    if discordant <= 0:
        return 0.0

    critical = _critical_value(discordant, alpha)

    if critical is None:
        return 0.0

    lower = sum(
        math.comb(discordant, k) * share ** k * (1.0 - share) ** (discordant - k)
        for k in range(critical + 1)
    )
    upper = sum(
        math.comb(discordant, k) * share ** k * (1.0 - share) ** (discordant - k)
        for k in range(discordant - critical, discordant + 1)
    )

    return lower + upper


@dataclass(frozen=True)
class PowerResult:
    discordant_rate: float
    older_share: float
    alpha: float
    power: float
    discordant_pairs_needed: int
    total_pairs_needed: int
    candidates_needed: Optional[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "discordant_rate": self.discordant_rate,
            "older_share": self.older_share,
            "alpha": self.alpha,
            "power": self.power,
            "discordant_pairs_needed": self.discordant_pairs_needed,
            "total_pairs_needed": self.total_pairs_needed,
            "candidates_needed": self.candidates_needed,
        }


def required_pairs(
    *,
    discordant_rate: Optional[float],
    older_share: Optional[float],
    alpha: float = 0.05,
    power: float = 0.80,
    yield_rate: Optional[float] = None,
) -> PowerResult:
    """Usable pairs needed to detect the given admission asymmetry.

    Args:
        discordant_rate: measured share of pairs with differing admission
            decisions. Required - it cannot be assumed.
        older_share: measured share of discordant pairs favouring the older
            passage. Required. 0.5 means no asymmetry and has no finite
            sample size.
        yield_rate: pilot yield (eligible / candidates). When given, the
            result also reports how many candidates must be constructed.

    Raises:
        ValueError: if either estimate is missing or degenerate.
    """

    if discordant_rate is None or older_share is None:
        raise ValueError(
            "discordant_rate and older_share must be measured on the pilot. "
            "They are not given defaults: a sample size computed from an "
            "assumed effect size is not a power analysis."
        )

    if not 0.0 < discordant_rate <= 1.0:
        raise ValueError("discordant_rate must be in (0, 1].")

    if not 0.0 <= older_share <= 1.0:
        raise ValueError("older_share must be in [0, 1].")

    if abs(older_share - 0.5) < 1e-9:
        raise ValueError(
            "older_share of 0.5 is exactly the null hypothesis; no finite "
            "sample size can detect it. Report the pilot estimate and its "
            "confidence interval instead."
        )

    if alpha not in _ALPHAS:
        raise ValueError(f"alpha must be one of {sorted(_ALPHAS)}.")

    if power not in _POWERS:
        raise ValueError(f"power must be one of {sorted(_POWERS)}.")

    # Smallest discordant count whose exact power reaches the target. Power
    # is not perfectly monotone in the count (the critical value moves in
    # steps), so the first crossing is confirmed to hold rather than assumed.
    discordant_needed = None
    for candidate in range(1, _MAX_DISCORDANT + 1):
        if exact_power(candidate, older_share, alpha) >= power:
            discordant_needed = candidate
            break

    if discordant_needed is None:
        raise ValueError(
            f"no discordant count below {_MAX_DISCORDANT} reaches "
            f"{power:.0%} power at older_share={older_share}."
        )

    total_needed = math.ceil(discordant_needed / discordant_rate)

    candidates = None
    if yield_rate is not None:
        if not 0.0 < yield_rate <= 1.0:
            raise ValueError("yield_rate must be in (0, 1].")
        candidates = math.ceil(total_needed / yield_rate)

    return PowerResult(
        discordant_rate=discordant_rate,
        older_share=older_share,
        alpha=alpha,
        power=power,
        discordant_pairs_needed=discordant_needed,
        total_pairs_needed=total_needed,
        candidates_needed=candidates,
    )


def detectable_effect(
    *,
    total_pairs: int,
    discordant_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Optional[float]:
    """Smallest older-favouring share detectable with ``total_pairs``.

    The inverse of :func:`required_pairs`, for the situation this thesis is
    actually in: the pool is fixed by the external dataset rather than chosen,
    so the honest question is not "how many pairs do we need?" but "what could
    this many pairs detect?". Returns None when no share in (0.5, 1] suffices.

    Solved by bisection on the monotone relationship between the share and
    the discordant pairs required; a closed form exists but is less legible
    and no more accurate here.
    """

    if total_pairs <= 0:
        raise ValueError("total_pairs must be positive.")
    if not 0.0 < discordant_rate <= 1.0:
        raise ValueError("discordant_rate must be in (0, 1].")
    if alpha not in _ALPHAS:
        raise ValueError(f"alpha must be one of {sorted(_ALPHAS)}.")
    if power not in _POWERS:
        raise ValueError(f"power must be one of {sorted(_POWERS)}.")

    available = int(total_pairs * discordant_rate)

    if available <= 0 or exact_power(available, 1.0 - 1e-9, alpha) < power:
        return None

    # Exact power rises with the share, so bisect and then round the answer
    # UP to a value that genuinely clears the target.
    low, high = 0.5, 1.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if exact_power(available, mid, alpha) < power:
            low = mid
        else:
            high = mid

    share = math.ceil(high * 10000.0) / 10000.0

    return share if exact_power(available, share, alpha) >= power else None


def sensitivity_curve(
    *,
    total_pairs: int,
    discordant_rates: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
    alpha: float = 0.05,
    power: float = 0.80,
) -> list[dict[str, object]]:
    """What a fixed pool could detect across plausible discordant rates.

    Reported as a curve rather than a point because the discordant rate is a
    Stage-3 measurement. Assuming one value and quoting a single sample size
    would be the fabricated effect size this module exists to refuse.
    """

    return [
        {
            "discordant_rate": rate,
            "expected_discordant_pairs": round(total_pairs * rate, 1),
            "min_detectable_older_share": detectable_effect(
                total_pairs=total_pairs,
                discordant_rate=rate,
                alpha=alpha,
                power=power,
            ),
        }
        for rate in discordant_rates
    ]
